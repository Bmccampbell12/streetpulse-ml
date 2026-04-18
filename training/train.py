from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

from app.config import (
    BATCH_SIZE,
    CALIBRATION_PATH,
    CALIBRATION_TEMPERATURE_DEFAULT,
    CLASS_INDEX_PATH,
    EPOCHS,
    HARD_NEGATIVE_DIR,
    LABELED_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    RANDOM_SEED,
    TORCH_MODEL_PATH,
    USE_WEIGHTED_SAMPLER,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_train_transform() -> transforms.Compose:
    def _jpeg_like(x):
        # Light quantization/compression-like perturbation for edge device realism.
        arr = np.asarray(x, dtype=np.uint8)
        arr = ((arr // 16) * 16).astype(np.uint8)
        return arr

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.4),
            transforms.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.15, hue=0.03),
            transforms.RandomRotation(degrees=6, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomApply([transforms.Lambda(_jpeg_like)], p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _temperature_scale(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return CALIBRATION_TEMPERATURE_DEFAULT

    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=0.05), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.detach().clamp(min=0.05).item())


def _build_version_tag(weighted_sampler: bool, has_hard_negatives: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    mode = "weighted_sampler" if weighted_sampler else "weighted_loss"
    hard = "_hardneg" if has_hard_negatives else ""
    return f"v{stamp}_{mode}{hard}"


def _collect_targets(dataset: Dataset, class_count: int) -> list[int]:
    if isinstance(dataset, ConcatDataset):
        out: list[int] = []
        for ds in dataset.datasets:
            out.extend(_collect_targets(ds, class_count))
        return out

    targets = getattr(dataset, "targets", None)
    if isinstance(targets, list):
        return [int(x) for x in targets if 0 <= int(x) < class_count]
    return []


def train() -> dict[str, object]:
    _set_seed(RANDOM_SEED)

    train_transform = _build_train_transform()
    eval_transform = _build_eval_transform()

    # Verify required class folders have images before handing off to ImageFolder.
    missing: list[str] = []
    for label in ["crack", "normal", "pothole", "manhole"]:
        label_dir = LABELED_DIR / label
        has_images = label_dir.is_dir() and any(
            f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
            for f in label_dir.iterdir()
            if f.is_file()
        )
        if not has_images:
            missing.append(str(label_dir))

    if missing:
        lines = "\n  ".join(missing)
        raise ValueError(
            f"Training requires labeled images. Add images to these folders:\n  {lines}\n"
            "Use the Dataset Control Panel (http://localhost:5174) to stream and label images, "
            "or copy images in manually."
        )

    base_dataset = torchvision.datasets.ImageFolder(str(LABELED_DIR), transform=train_transform)
    eval_base_dataset = torchvision.datasets.ImageFolder(str(LABELED_DIR), transform=eval_transform)
    if len(base_dataset) == 0:
        raise ValueError("No labeled samples found in dataset/labeled")

    train_dataset: Dataset = base_dataset
    eval_dataset: Dataset = eval_base_dataset
    has_hard_negatives = False
    if HARD_NEGATIVE_DIR.exists() and any(HARD_NEGATIVE_DIR.rglob("*.*")):
        hard_train = torchvision.datasets.ImageFolder(str(HARD_NEGATIVE_DIR), transform=train_transform)
        hard_eval = torchvision.datasets.ImageFolder(str(HARD_NEGATIVE_DIR), transform=eval_transform)
        if len(hard_train) > 0:
            has_hard_negatives = True
            train_dataset = ConcatDataset([base_dataset, hard_train])
            eval_dataset = ConcatDataset([eval_base_dataset, hard_eval])

    targets = _collect_targets(train_dataset, NUM_CLASSES)
    if not targets:
        raise ValueError("Failed to collect class targets for training")

    # Keep at least one batch for validation for calibration.
    full_indices = torch.randperm(len(train_dataset)).tolist()
    val_size = max(1, int(0.1 * len(full_indices)))
    if val_size >= len(full_indices):
        val_size = max(1, len(full_indices) - 1)
    val_indices = full_indices[:val_size]
    train_indices = full_indices[val_size:]
    if not train_indices:
        train_indices = val_indices

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_dataset, val_indices)

    train_targets = [targets[i] for i in train_indices]
    class_counts = Counter(train_targets)

    # Inverse-frequency weights for class imbalance handling.
    weights = []
    for idx in range(NUM_CLASSES):
        count = class_counts.get(idx, 1)
        weights.append(len(train_targets) / (NUM_CLASSES * count))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    sampler = None
    shuffle = True
    if USE_WEIGHTED_SAMPLER:
        sample_weights = [weights[label] for label in train_targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=shuffle, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    best_model_path = TORCH_MODEL_PATH.parent / "best_model.pth"

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation loss for best checkpoint tracking
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (predictions == labels).sum().item()
        
        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {scheduler.get_last_lr()[0]:.6f}")

    model.eval()
    val_logits: list[torch.Tensor] = []
    val_labels: list[torch.Tensor] = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_logits.append(outputs)
            val_labels.append(labels.to(device))
            
            # Collect predictions and labels for classification report
            preds = outputs.argmax(dim=1).cpu().tolist()
            lbls = labels.tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

    logits = torch.cat(val_logits, dim=0) if val_logits else torch.empty(0, NUM_CLASSES, device=device)
    labels_tensor = torch.cat(val_labels, dim=0) if val_labels else torch.empty(0, dtype=torch.long, device=device)
    temperature = _temperature_scale(logits, labels_tensor)

    # Print per-class metrics
    class_names = [k for k, _ in sorted(base_dataset.class_to_idx.items(), key=lambda x: x[1])]
    print("\nPer-class Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    version = _build_version_tag(USE_WEIGHTED_SAMPLER, has_hard_negatives)
    version_weights_path = TORCH_MODEL_PATH.parent / f"{version}.pth"

    TORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), version_weights_path)
    
    # Copy best model to standard location if it exists
    if best_model_path.exists():
        import shutil
        shutil.copyfile(best_model_path, TORCH_MODEL_PATH)
    else:
        torch.save(model.state_dict(), TORCH_MODEL_PATH)
    
    CLASS_INDEX_PATH.write_text(json.dumps(base_dataset.class_to_idx, indent=2), encoding="utf-8")
    CALIBRATION_PATH.write_text(json.dumps({"temperature": temperature, "model_version": version}, indent=2), encoding="utf-8")

    return {
        "model_version": version,
        "temperature": temperature,
        "class_weights": weights,
        "hard_negatives": has_hard_negatives,
        "weighted_sampler": USE_WEIGHTED_SAMPLER,
        "weights_path": str(version_weights_path),
    }
