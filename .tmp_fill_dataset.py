from pathlib import Path
from PIL import Image

root = Path('dataset/labeled')
labels = ['crack', 'normal', 'pothole', 'manhole']

for label in labels:
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    images = [p for p in folder.iterdir() if p.is_file()]
    print(label, 'exists=', folder.exists(), 'files=', len(images))
    if not images:
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        path = folder / 'placeholder.png'
        img.save(path)
        print('  created', path)
