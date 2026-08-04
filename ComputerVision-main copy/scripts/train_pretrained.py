from pathlib import Path
import csv
import sys
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


NUMBER_OF_CLASSES = 500
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01


def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_images += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    loss = total_loss / total_images
    acc = total_correct / total_images
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    return loss, acc, f1


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_transform = train_transform
    
    train_dataset = INaturalistSubset(
        csv_path=PROJECT_ROOT / "splits" / "train.csv",
        project_root=PROJECT_ROOT,
        transform=train_transform,
    )

    val_dataset = INaturalistSubset(
        csv_path=PROJECT_ROOT / "splits" / "validation.csv",
        project_root=PROJECT_ROOT,
        transform=val_transform,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("Loading pretrained ResNet-18...")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, NUMBER_OF_CLASSES)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    output_dir = PROJECT_ROOT / "outputs"

    checkpoint_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    best_val_f1 = -1.0
    best_checkpoint_path = checkpoint_dir / "pretrained_frozen_best.pth"

    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        # Keep the frozen backbone and BatchNorm layers in evaluation mode.
        # Only the newly created classification layer is trained.
        model.eval()
        model.fc.train()

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for batch_num, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_images += labels.size(0)

            if batch_num % 100 == 0:
                print(f"Epoch {epoch + 1}: batch {batch_num}/{len(train_loader)}")

        train_loss = total_loss / total_images
        train_acc = total_correct / total_images

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_function, device)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {train_loss:.4f}")
        print(f"Training accuracy: {train_acc * 100:.2f}%")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc * 100:.2f}%")
        print(f"Validation macro-F1: {val_f1:.4f}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch time: {epoch_time:.1f} seconds")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "epoch_seconds": epoch_time,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_f1": val_f1,
                    "validation_accuracy": val_acc,
                    "number_of_classes": NUMBER_OF_CLASSES,
                    "image_size": 128,
                },
                best_checkpoint_path,
            )

            print(f"Saved new best model with macro-F1: {val_f1:.4f}")

    checkpoint_path = checkpoint_dir / "pretrained_frozen_final.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "number_of_classes": NUMBER_OF_CLASSES,
            "image_size": 128,
            "epochs": EPOCHS,
        },
        checkpoint_path,
    )

    history_path = output_dir / "pretrained_frozen_history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    elapsed_time = time.time() - start_time

    print(f"\nFinished in {elapsed_time:.1f} seconds.")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"History saved to: {history_path}")
    print("500-class frozen pretrained training completed.")


if __name__ == "__main__":
    main()