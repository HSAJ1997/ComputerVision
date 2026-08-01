from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


class RemappedSubset(Dataset):
    def __init__(self, dataset, selected_classes):
        self.dataset = dataset
        self.class_mapping = {
            old_label: new_label
            for new_label, old_label in enumerate(selected_classes)
        }
        self.indices = []

        for i, row in enumerate(dataset.samples):
            label = int(row["class_index"])
            if label in self.class_mapping:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_index = self.indices[index]
        image, label = self.dataset[real_index]
        new_label = self.class_mapping[label]
        return image, new_label


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_images += labels.size(0)

    return total_loss / total_images, total_correct / total_images


def main():
    torch.manual_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = INaturalistSubset(
        csv_path=PROJECT_ROOT / "splits" / "train.csv",
        project_root=PROJECT_ROOT,
        transform=train_transform,
    )

    full_val = INaturalistSubset(
        csv_path=PROJECT_ROOT / "splits" / "validation.csv",
        project_root=PROJECT_ROOT,
        transform=val_transform,
    )

    available_classes = sorted({int(row["class_index"]) for row in full_train.samples})
    selected_classes = available_classes[:10]

    print(f"Selected classes: {selected_classes}")

    train_dataset = RemappedSubset(full_train, selected_classes)
    val_dataset = RemappedSubset(full_val, selected_classes)

    print(f"Sanity train size: {len(train_dataset)}")
    print(f"Sanity val size: {len(val_dataset)}")

    assert len(train_dataset) == 400
    assert len(val_dataset) == 100

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    epochs = 3
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_images += labels.size(0)

        train_loss = total_loss / total_images
        train_acc = total_correct / total_images

        val_loss, val_acc = evaluate(model, val_loader, loss_function, device)

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_acc * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc * 100:.2f}%")

    elapsed_time = time.time() - start_time

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "sanity_resnet18.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_classes": selected_classes,
            "number_of_classes": 10,
        },
        checkpoint_path,
    )

    print(f"\nTraining finished in {elapsed_time:.1f} seconds.")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("Sanity training check passed.")


if __name__ == "__main__":
    main()