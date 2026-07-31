import os

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
)
from dataset_loading import buildDataLoaders
from device import getDevice
from model import buildResnet18


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
# model generalizes to images it was not trained on.
def validateOneEpoch(model, loader, lossFunction, device):
    model.eval()

    totalLoss = 0.0
    correctCount = 0
    totalCount = 0

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

    averageLoss = totalLoss / totalCount
    accuracy = correctCount / totalCount
    return averageLoss, accuracy


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=BATCH_SIZE, numWorkers=NUM_WORKERS
    )

    model = buildResnet18(NUM_CLASSES)
    model = model.to(device)

    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    bestValidationAccuracy = 0.0

    epoch = 1
    while epoch <= NUM_EPOCHS:
        trainLoss, trainAccuracy = trainOneEpoch(model, trainLoader, lossFunction, optimizer, device)
        validationLoss, validationAccuracy = validateOneEpoch(model, validationLoader, lossFunction, device)

        print("epoch", epoch, "of", NUM_EPOCHS)
        print("  train loss:", trainLoss, "train accuracy:", trainAccuracy)
        print("  validation loss:", validationLoss, "validation accuracy:", validationAccuracy)

        if validationAccuracy > bestValidationAccuracy:
            bestValidationAccuracy = validationAccuracy
            torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
            print("  saved new best checkpoint")

        epoch = epoch + 1
