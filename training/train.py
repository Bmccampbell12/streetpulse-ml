from __future__ import annotations

import json
import random

import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from app.config import (
    BATCH_SIZE,
    CLASS_INDEX_PATH,
    EPOCHS,
    LABELED_DIR,
    LEARNING_RATE,
    NUM_CLASSES,
    RANDOM_SEED,
    TORCH_MODEL_PATH,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train() -> None:
    _set_seed(RANDOM_SEED)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.ImageFolder(str(LABELED_DIR), transform=transform)
    if len(dataset) == 0:
        raise ValueError("No labeled samples found in dataset/labeled")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
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
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    TORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), TORCH_MODEL_PATH)
    CLASS_INDEX_PATH.write_text(json.dumps(dataset.class_to_idx, indent=2), encoding="utf-8")
