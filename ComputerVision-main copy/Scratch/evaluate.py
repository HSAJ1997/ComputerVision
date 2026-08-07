import csv
import time
import os

import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score

from config import (
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    BEST_CHECKPOINT_PATH,
    PROJECT_ROOT
)
from dataset_loading import buildDataLoaders
from device import getDevice
from model import buildResnet18

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "scratch_test_predictions_aug_step.csv")

# Runs the trained model once over the test set. For every image it records
# the true label, the top-1 prediction, and the five highest-scoring class
# guesses (needed for top-5 accuracy and for later confusion-matrix / failure
# analysis). Returns everything needed to compute the full metric set, plus
# the total inference time.
def runTestInference(model, loader, device):
    model.eval()

    allLabels = []
    allTop1 = []
    allTop5 = []

    inferenceStartTime = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Top-5 predicted class indices for each image, highest score first.
            top5 = torch.topk(outputs, k=5, dim=1).indices
            top1 = top5[:, 0]

            allLabels.extend(labels.cpu().tolist())
            allTop1.extend(top1.cpu().tolist())
            allTop5.extend(top5.cpu().tolist())

    inferenceSeconds = time.time() - inferenceStartTime
    return allLabels, allTop1, allTop5, inferenceSeconds


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=BATCH_SIZE, numWorkers=NUM_WORKERS
    )

    model = buildResnet18(NUM_CLASSES)

    # The checkpoint is a dict (matching the pretrained model's format), so the
    # model weights live under "model_state_dict".
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    labels, top1, top5, inferenceSeconds = runTestInference(model, testLoader, device)

    totalImages = len(labels)

    # Top-1 accuracy: the single best guess is correct.
    top1Correct = 0
    for i in range(totalImages):
        if top1[i] == labels[i]:
            top1Correct = top1Correct + 1
    top1Accuracy = top1Correct / totalImages

    # Top-5 accuracy: the true label is anywhere in the five best guesses.
    top5Correct = 0
    for i in range(totalImages):
        if labels[i] in top5[i]:
            top5Correct = top5Correct + 1
    top5Accuracy = top5Correct / totalImages

    # Macro-averaged precision / recall / F1: computed per class, then plain-
    # averaged, so every species counts equally regardless of how common it is.
    macroPrecision = precision_score(labels, top1, average="macro", zero_division=0)
    macroRecall = recall_score(labels, top1, average="macro", zero_division=0)
    macroF1 = f1_score(labels, top1, average="macro", zero_division=0)

    print("")
    print("test images:", totalImages)
    print("top-1 accuracy:", top1Accuracy)
    print("top-5 accuracy:", top5Accuracy)
    print("macro precision:", macroPrecision)
    print("macro recall:", macroRecall)
    print("macro F1:", macroF1)
    print("test inference seconds:", inferenceSeconds)

    # Save per-image predictions so the confusion matrix, most-confused pairs,
    # and failure-case images can all be built later without re-running the model.
    with open(PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_index", "true_label", "predicted_label", "top5_labels"])
        for i in range(totalImages):
            top5String = " ".join(str(c) for c in top5[i])
            writer.writerow([i, labels[i], top1[i], top5String])

    print("predictions saved to:", PREDICTIONS_PATH)
