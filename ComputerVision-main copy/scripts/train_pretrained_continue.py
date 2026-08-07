from pathlib import Path
import csv
import sys
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


NUM_CLASSES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 32
EXTRA_EPOCHS = 3
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
# EARLY_STOPPING_PATIENCE = 2
RANDOM_SEED = 42

START_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "pretrained_finetuned_best.pth"
)


def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    labels_list = []
    predictions_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_images += labels.size(0)

            labels_list.extend(labels.cpu().tolist())
            predictions_list.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    macro_f1 = f1_score(
        labels_list,
        predictions_list,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


def set_finetuning_mode(model):
    # Keep the frozen part of the network in evaluation mode.
    model.train()

    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    model.layer4.train()
    model.fc.train()


def main():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda")
    print(f"Continuing fine-tuning on {torch.cuda.get_device_name(0)}.")

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {START_CHECKPOINT}"
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.5, 1.0),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Keep the AdamW state from the first fine-tuning stage.
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = LEARNING_RATE
            parameter_group["weight_decay"] = WEIGHT_DECAY

    starting_epoch = checkpoint.get("epoch", 3)
    best_val_f1 = checkpoint.get("validation_macro_f1", 0.4180)

    print(
        f"Starting from epoch {starting_epoch} "
        f"with validation F1 {best_val_f1:.4f}."
    )
    print(f"New learning rate: {LEARNING_RATE}")

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    output_dir = PROJECT_ROOT / "outputs"

    checkpoint_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    best_checkpoint_path = (
        checkpoint_dir / "pretrained_finetuned_stage2_best.pth"
    )
    final_checkpoint_path = (
        checkpoint_dir / "pretrained_finetuned_stage2_final.pth"
    )
    history_path = (
        output_dir / "pretrained_finetuned_stage2_history.csv"
    )

    history = []
    best_epoch = starting_epoch
    epochs_without_improvement = 0
    final_epoch = starting_epoch
    start_time = time.time()

    for local_epoch in range(1, EXTRA_EPOCHS + 1):
        epoch_number = starting_epoch + local_epoch
        final_epoch = epoch_number
        epoch_start = time.time()

        set_finetuning_mode(model)

        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for batch_num, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (
                predictions == labels
            ).sum().item()
            total_images += labels.size(0)

            if batch_num % 100 == 0:
                print(
                    f"Epoch {epoch_number}: "
                    f"batch {batch_num}/{len(train_loader)}"
                )

        train_loss = total_loss / total_images
        train_accuracy = total_correct / total_images

        val_loss, val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )

        epoch_seconds = time.time() - epoch_start

        print(
            f"Epoch {epoch_number} finished in {epoch_seconds:.1f}s | "
            f"train acc: {train_accuracy * 100:.2f}% | "
            f"val acc: {val_accuracy * 100:.2f}% | "
            f"val F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch_number,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
            "epoch_seconds": epoch_seconds,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch_number
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch_number,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_f1": val_f1,
                    "validation_accuracy": val_accuracy,
                    "number_of_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "fine_tuned_layers": ["layer4", "fc"],
                    "starting_checkpoint": START_CHECKPOINT.name,
                },
                best_checkpoint_path,
            )

            print("Saved a new best model.")
        # else:
        #     epochs_without_improvement += 1
        #     print(
        #         "Validation F1 did not improve "
        #         f"({epochs_without_improvement}/"
        #         f"{EARLY_STOPPING_PATIENCE})."
        #     )

        #     if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        #         print("Stopping early.")
        #         break

    torch.save(
        {
            "epoch": final_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "number_of_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "fine_tuned_layers": ["layer4", "fc"],
            "starting_checkpoint": START_CHECKPOINT.name,
        },
        final_checkpoint_path,
    )

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    total_time = time.time() - start_time

    print(
        f"Stage 2 finished in {total_time:.1f}s. "
        f"Best epoch: {best_epoch}, "
        f"best validation F1: {best_val_f1:.4f}."
    )

    if best_epoch == starting_epoch:
        print(
            "No stage 2 epoch improved on the original checkpoint. "
            "Keep pretrained_finetuned_best.pth."
        )
    else:
        print(
            "Use pretrained_finetuned_stage2_best.pth "
            "for the final comparison."
        )


if __name__ == "__main__":
    main()
