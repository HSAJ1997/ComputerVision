import os
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    BEST_CHECKPOINT_PATH,
    PROJECT_ROOT,
    USE_AUGMENTATION,
)
from dataset_loading import buildDataLoaders
from device import getDevice
from model import buildResnet18
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import f1_score

# Reuse the exact same train/validate functions you already wrote.
from train import trainOneEpoch, validateOneEpoch, setSeed

# ---- how many MORE epochs to run, on top of what the checkpoint already did ----
ADDITIONAL_EPOCHS = 10
# LR to resume with. Cosine had decayed to ~0 by epoch 30, so pick a small but
# non-zero value or the extra epochs do almost nothing. 1e-3 is a reasonable
# "fine-tune a bit more" rate; raise toward LEARNING_RATE if you want bigger moves.
RESUME_LEARNING_RATE = 1e-3

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
# NEW filenames so you don't overwrite the original run's history.
HISTORY_PATH = os.path.join(OUTPUT_DIR, "scratch_history_aug_step_resumed.csv")
# Where to save the continued checkpoint. Keep it separate so you can always
# fall back to the original best if resuming makes things worse.
RESUMED_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_scratch_resumed_best.pth")


if __name__ == "__main__":
    setSeed(42)
    device = getDevice()
    print("using device:", device)

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=BATCH_SIZE, numWorkers=NUM_WORKERS, useAugmentation=USE_AUGMENTATION
    )

    model = buildResnet18(NUM_CLASSES)
    model = model.to(device)

    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=RESUME_LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # ---- load the checkpoint: weights AND optimizer state ----
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # load_state_dict restores the OLD lr into the optimizer, so force the
    # resume lr back in on every param group.
    for group in optimizer.param_groups:
        group["lr"] = RESUME_LEARNING_RATE

    startEpoch = checkpoint["epoch"] + 1                 # e.g. 31
    endEpoch = checkpoint["epoch"] + ADDITIONAL_EPOCHS   # e.g. 40
    bestValidationF1 = checkpoint.get("validation_macro_f1", -1.0)
    print(f"resuming from epoch {checkpoint['epoch']} "
          f"(best macro-F1 so far: {bestValidationF1:.4f})")

    # Fresh cosine schedule just over the additional epochs.
    scheduler = CosineAnnealingLR(optimizer, T_max=ADDITIONAL_EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = []
    trainingStartTime = time.time()

    epoch = startEpoch
    while epoch <= endEpoch:
        epochStartTime = time.time()

        trainLoss, trainAccuracy = trainOneEpoch(model, trainLoader, lossFunction, optimizer, device)
        validationLoss, validationAccuracy, validationMacroF1 = validateOneEpoch(model, validationLoader, lossFunction, device)

        epochSeconds = time.time() - epochStartTime

        print("epoch", epoch, "of", endEpoch)
        print("  train loss:", trainLoss, "train accuracy:", trainAccuracy)
        print("  validation loss:", validationLoss, "validation accuracy:", validationAccuracy)
        print("  validation macro-F1:", validationMacroF1)
        print("  epoch seconds:", epochSeconds)

        history.append({
            "epoch": epoch,
            "train_loss": trainLoss,
            "train_accuracy": trainAccuracy,
            "validation_loss": validationLoss,
            "validation_accuracy": validationAccuracy,
            "validation_macro_f1": validationMacroF1,
            "epoch_seconds": epochSeconds,
        })

        if validationMacroF1 > bestValidationF1:
            bestValidationF1 = validationMacroF1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_macro_f1": validationMacroF1,
                    "validation_accuracy": validationAccuracy,
                    "number_of_classes": NUM_CLASSES,
                    "image_size": 224,
                },
                RESUMED_CHECKPOINT_PATH,
            )
            print("  saved new best checkpoint (macro-F1:", validationMacroF1, ")")

        epoch = epoch + 1

        scheduler.step()
        print("  next-epoch learning rate:", scheduler.get_last_lr()[0])

    totalTrainingSeconds = time.time() - trainingStartTime

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("")
    print("total additional training seconds:", totalTrainingSeconds)
    print("best validation macro-F1 (overall):", bestValidationF1)
    print("history saved to:", HISTORY_PATH)