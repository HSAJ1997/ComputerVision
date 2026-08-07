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
    NUM_EPOCHS,
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


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "scratch_history_baseline_adam_optimizer.csv")

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def setSeed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Runs one full pass over the training data. For every batch it makes
# a prediction, measures how wrong it was, and updates the model's
# weights to do better next time. Returns the average loss and
# accuracy across the whole epoch.
def trainOneEpoch(model, loader, lossFunction, optimizer, device):
    model.train()

    totalLoss = 0.0
    correctCount = 0
    totalCount = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = lossFunction(outputs, labels)
        loss.backward()
        optimizer.step()

        totalLoss = totalLoss + loss.item() * images.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correctCount = correctCount + (predictions == labels).sum().item()
        totalCount = totalCount + images.size(0)

    averageLoss = totalLoss / totalCount
    accuracy = correctCount / totalCount
    return averageLoss, accuracy


# Runs one full pass over the validation data without changing any
# weights. Used after every training epoch to check how well the
# model generalizes to images it was not trained on. Also collects all
# labels and predictions so macro-averaged F1 can be computed across

# the 500 classes.
def validateOneEpoch(model, loader, lossFunction, device):
    model.eval()

    totalLoss = 0.0
    correctCount = 0
    totalCount = 0

    allLabels = []
    allPredictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = lossFunction(outputs, labels)

            totalLoss = totalLoss + loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correctCount = correctCount + (predictions == labels).sum().item()
            totalCount = totalCount + images.size(0)

            allLabels.extend(labels.cpu().tolist())
            allPredictions.extend(predictions.cpu().tolist())

    averageLoss = totalLoss / totalCount
    accuracy = correctCount / totalCount
    macroF1 = f1_score(allLabels, allPredictions, average="macro", zero_division=0)
    return averageLoss, accuracy, macroF1


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
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bestValidationF1 = -1.0

    # Record
    history = []
    trainingStartTime = time.time()

    epoch = 1
    while epoch <= NUM_EPOCHS:
        epochStartTime = time.time()

        trainLoss, trainAccuracy = trainOneEpoch(model, trainLoader, lossFunction, optimizer, device)
        validationLoss, validationAccuracy, validationMacroF1  = validateOneEpoch(model, validationLoader, lossFunction, device)

        epochSeconds = time.time() - epochStartTime

        clear()
        print("epoch", epoch, "of", NUM_EPOCHS)
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
                BEST_CHECKPOINT_PATH,
            )

            print("  saved new best checkpoint (macro-F1:", validationMacroF1, ")")

        epoch = epoch + 1

        scheduler.step()   
        currentLr = scheduler.get_last_lr()[0]
        print("  next-epoch learning rate:", currentLr)
    
    totalTrainingSeconds = time.time() - trainingStartTime

    # Write the history to CSV. Column names match the pretrained
    # model's outputs/pretrained_frozen_history.csv
    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())

        writer.writeheader()
        writer.writerows(history)
 
    print("")
    print("total training seconds:", totalTrainingSeconds)
    print("best validation macro-F1:", bestValidationF1)
    print("history saved to:", HISTORY_PATH)
