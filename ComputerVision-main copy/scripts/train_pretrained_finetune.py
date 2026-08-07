from pathlib import Path
import csv
import sys
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


NUM_CLASSES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
MINIMUM_LEARNING_RATE = 0.0
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
RANDOM_SEED = 42
USE_AUGMENTATION = True

# Augmentation strength. Keep these identical to the frozen-backbone script,
# otherwise the two runs differ in more than one variable and the comparison
# between them says nothing.
CROP_SCALE = (0.7, 1.0)
HORIZONTAL_FLIP_PROBABILITY = 0.5
JITTER_BRIGHTNESS = 0.2
JITTER_CONTRAST = 0.2
JITTER_SATURATION = 0.2
JITTER_HUE = 0.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

START_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "pretrained_frozen_aug_step_adam_best.pth"
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
    # Train layer4 and fc, but keep the frozen backbone in evaluation mode.
    # This stops the frozen BatchNorm statistics from changing.
    model.train()

    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    model.layer4.train()
    model.fc.train()


def build_train_transform():
    if USE_AUGMENTATION:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                (IMAGE_SIZE, IMAGE_SIZE),
                scale=CROP_SCALE,
            ),
            transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
            transforms.ColorJitter(
                brightness=JITTER_BRIGHTNESS,
                contrast=JITTER_CONTRAST,
                saturation=JITTER_SATURATION,
                hue=JITTER_HUE,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda")
    print("Using GPU for fine-tuning.")
    print(f"Augmentation enabled: {USE_AUGMENTATION}")

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {START_CHECKPOINT}"
        )

    train_transform = build_train_transform()
    val_transform = build_eval_transform()

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
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    print(
        f"Loaded {len(train_dataset)} training images and "
        f"{len(val_dataset)} validation images."
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu")

    # Check the class count before loading the weights. Doing it afterwards is
    # pointless: load_state_dict already raises a shape-mismatch error on the
    # final layer, so this message would never be reached.
    checkpoint_classes = checkpoint.get("number_of_classes")
    if checkpoint_classes is not None and checkpoint_classes != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint has {checkpoint_classes} classes, "
            f"but this script expects {NUM_CLASSES}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded the frozen model from epoch "
        f"{checkpoint.get('epoch', 'unknown')}."
    )

    # Freeze the whole model first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final residual block and the classifier.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    # Pass only the parameters that are actually being trained. Handing over
    # every parameter happens to work, because frozen tensors never receive a
    # gradient and SGD skips them, but it hides which layers are unfrozen and
    # makes the saved optimizer state describe 62 parameters instead of the
    # 22 that this run updates.
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"Training {len(trainable_parameters)} parameter tensors "
        f"(layer4 and fc)."
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, momentum=0.9
    )

    # Cosine annealing across the whole run, stepped once per epoch, matching
    # the frozen-backbone script. This matters more here than there: 35 epochs
    # at a constant 0.01 would keep making large updates to pretrained layer4
    # weights long after they should have settled.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MINIMUM_LEARNING_RATE,
    )

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    output_dir = PROJECT_ROOT / "outputs"

    checkpoint_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    best_checkpoint_path = (
        checkpoint_dir / "pretrained_finetuned_aug_step_best_adam.pth"
    )
    final_checkpoint_path = (
        checkpoint_dir / "pretrained_finetuned_aug_step_final_adam.pth"
    )
    history_path = (
        output_dir / "pretrained_finetuned_aug_step_adam_history.csv"
    )

    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        set_finetuning_mode(model)

        # The learning rate used by every batch in this epoch, read before the
        # scheduler steps at the end of the loop.
        current_learning_rate = optimizer.param_groups[0]["lr"]

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
                    f"Epoch {epoch + 1}/{EPOCHS}: "
                    f"batch {batch_num}/{len(train_loader)}"
                )

        train_loss = total_loss / total_images
        train_acc = total_correct / total_images

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )

        # Step once per epoch, after training and validation are finished.
        scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{EPOCHS} finished in "
            f"{epoch_time:.1f}s | "
            f"lr: {current_learning_rate:.6f} | "
            f"train acc: {train_acc * 100:.2f}% | "
            f"val acc: {val_acc * 100:.2f}% | "
            f"val F1: {val_f1:.4f}"
        )
        print(f"Next epoch learning rate: {next_learning_rate:.6f}")

        history.append({
            "epoch": epoch + 1,
            "learning_rate": current_learning_rate,
            "next_learning_rate": next_learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "epoch_seconds": epoch_time,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_macro_f1": val_f1,
                    "validation_accuracy": val_acc,
                    "number_of_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE,
                    "learning_rate": current_learning_rate,
                    "initial_learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "momentum": MOMENTUM,
                    "use_augmentation": USE_AUGMENTATION,
                    "scheduler": "CosineAnnealingLR",
                    "fine_tuned_layers": ["layer4", "fc"],
                    "starting_checkpoint": START_CHECKPOINT.name,
                },
                best_checkpoint_path,
            )

            print("Saved a new best model.")

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "number_of_classes": NUM_CLASSES,
            "image_size": IMAGE_SIZE,
            "epochs": EPOCHS,
            "initial_learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "momentum": MOMENTUM,
            "use_augmentation": USE_AUGMENTATION,
            "scheduler": "CosineAnnealingLR",
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
        f"Fine-tuning finished in {total_time:.1f}s. "
        f"Best epoch: {best_epoch}, "
        f"best validation F1: {best_val_f1:.4f}."
    )
    print(f"Saved results to {checkpoint_dir} and {output_dir}.")


if __name__ == "__main__":
    main()