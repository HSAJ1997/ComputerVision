from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = INaturalistSubset(
        csv_path=PROJECT_ROOT / "splits" / "train.csv",
        project_root=PROJECT_ROOT,
        transform=transform,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for img, label, ax in zip(images, labels, axes.flat):
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f"Class index: {label.item()}")
        ax.axis("off")

    plt.tight_layout()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "sample_batch.png"
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"Saved image grid to: {output_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()