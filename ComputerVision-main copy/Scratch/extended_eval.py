import csv
import json
import time
import os

import matplotlib
matplotlib.use("Agg")  # No display on the training machine, so render to file only.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)

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

# Every output shares the same "_aug_step" tag so this run's files never
# overwrite the pretrained model's results in the same folder.
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "scratch_test_predictions_adam_optimizer.csv")
METRICS_PATH = os.path.join(OUTPUT_DIR, "scratch_test_metrics_adam_optimizer.json")
PER_CLASS_PATH = os.path.join(OUTPUT_DIR, "scratch_per_class_metrics_adam_optimizer.csv")
MATRIX_CSV_PATH = os.path.join(OUTPUT_DIR, "scratch_confusion_matrix_adam_optimizer.csv")
MATRIX_IMAGE_PATH = os.path.join(OUTPUT_DIR, "scratch_confusion_matrix_adam_optimizer.png")
CONFUSIONS_PATH = os.path.join(OUTPUT_DIR, "scratch_common_confusions_adam_optimizer.csv")
CORRECT_GRID_PATH = os.path.join(OUTPUT_DIR, "scratch_correct_examples_adam_optimizer.png")
INCORRECT_GRID_PATH = os.path.join(OUTPUT_DIR, "scratch_incorrect_examples_adam_optimizer.png")

CLASS_FILE = os.path.join(PROJECT_ROOT, "splits", "selected_classes.json")

GRID_COLUMNS = 4
GRID_ROWS = 3
GRID_SIZE = GRID_COLUMNS * GRID_ROWS
TOP_CONFUSIONS_TO_SAVE = 50


# Species names are only used to label the figures. If the class file is not
# there the figures fall back to bare class indices, which is still readable.
def loadSpeciesNames():
    if not os.path.isfile(CLASS_FILE):
        return None

    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        classData = json.load(file)

    names = {}
    for item in classData["classes"]:
        names[item["class_index"]] = item["species_name"]

    if len(names) != NUM_CLASSES:
        return None

    return names


def describeClass(classIndex, speciesNames):
    if speciesNames is None:
        return "class " + str(classIndex)
    return speciesNames[classIndex]


# The example grids need the file on disk for each test image. Different
# dataset classes store this differently, so try the usual layouts and give up
# cleanly if none of them fit.
def getImagePaths(dataset, totalImages):
    candidate = getattr(dataset, "samples", None)
    if candidate is None:
        candidate = getattr(dataset, "imagePaths", None)
    if candidate is None:
        candidate = getattr(dataset, "image_paths", None)
    if candidate is None:
        return None

    paths = []
    for item in candidate:
        if isinstance(item, dict):
            paths.append(item.get("image_path"))
        elif isinstance(item, (list, tuple)):
            paths.append(item[0])
        else:
            paths.append(item)

    if len(paths) != totalImages:
        return None
    for path in paths:
        if path is None:
            return None

    return paths


# Rows are normalised so each one sums to 1. Without this the plot is
# dominated by whichever classes happen to have the most test images, and
# every row would need to be read against a different scale.
def saveConfusionMatrix(matrix, outputPath):
    rowTotals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        rowTotals,
        out=np.zeros_like(matrix, dtype=float),
        where=rowTotals != 0,
    )

    plt.figure(figsize=(16, 14))
    plt.imshow(normalised, interpolation="nearest", aspect="auto")
    plt.title("Normalised confusion matrix: ResNet-18 trained from scratch")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.colorbar(label="Proportion of true class")
    plt.tight_layout()
    plt.savefig(outputPath, dpi=200)
    plt.close()


def saveExampleGrid(records, outputPath, title):
    if not records:
        print("no records to plot for:", outputPath)
        return

    records = records[:GRID_SIZE]

    figure, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=(14, 11))
    axes = axes.flatten()

    for axis, record in zip(axes, records):
        imagePath = record["imagePath"]
        if not os.path.isabs(imagePath):
            imagePath = os.path.join(PROJECT_ROOT, imagePath)

        with Image.open(imagePath) as image:
            axis.imshow(image.convert("RGB"))

        axis.set_title(
            "True: " + record["trueName"] + "\n"
            + "Pred: " + record["predictedName"] + "\n"
            + "Confidence: " + format(record["top1Confidence"], ".3f"),
            fontsize=8,
        )
        axis.axis("off")

    # Blank out any unused cells when there were fewer than 12 records.
    for axis in axes[len(records):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=180)
    plt.close(figure)


# Runs the trained model once over the test set. For every image it records
# the true label, the top-1 prediction, and the five highest-scoring class
# guesses (needed for top-5 accuracy and for later confusion-matrix / failure
# analysis). Softmax probabilities are kept alongside the class indices so the
# per-image confidences can be written out and used to rank failure cases.
# Returns everything needed to compute the full metric set, plus the total
# inference time.
def runTestInference(model, loader, device):
    model.eval()

    allLabels = []
    allTop1 = []
    allTop5 = []
    allTop1Confidence = []
    allTop5Confidence = []

    inferenceStartTime = time.time()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Softmax is monotonic, so ranking by probability gives exactly the
            # same order as ranking by raw logit. Taking topk on the
            # probabilities just means the confidence values come out directly.
            probabilities = torch.softmax(outputs, dim=1)
            top5Probabilities, top5 = torch.topk(probabilities, k=5, dim=1)
            top1 = top5[:, 0]

            allLabels.extend(labels.cpu().tolist())
            allTop1.extend(top1.cpu().tolist())
            allTop5.extend(top5.cpu().tolist())
            allTop1Confidence.extend(top5Probabilities[:, 0].cpu().tolist())
            allTop5Confidence.extend(top5Probabilities.cpu().tolist())

    inferenceSeconds = time.time() - inferenceStartTime
    return (
        allLabels,
        allTop1,
        allTop5,
        allTop1Confidence,
        allTop5Confidence,
        inferenceSeconds,
    )


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=BATCH_SIZE, numWorkers=NUM_WORKERS
    )

    model = buildResnet18(NUM_CLASSES)

    # The checkpoint is a dict (matching the pretrained model's format), so the
    # model weights live under "model_state_dict".
    checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    (
        labels,
        top1,
        top5,
        top1Confidence,
        top5Confidence,
        inferenceSeconds,
    ) = runTestInference(model, testLoader, device)

    totalImages = len(labels)
    classLabels = list(range(NUM_CLASSES))
    speciesNames = loadSpeciesNames()

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

    # The same three quantities before averaging, one row per class. Passing
    # labels=classLabels forces all 500 classes to appear even if a class was
    # never predicted and never appeared in the test set.
    perClassPrecision, perClassRecall, perClassF1, support = precision_recall_fscore_support(
        labels,
        top1,
        labels=classLabels,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(labels, top1, labels=classLabels)

    print("")
    print("test images:", totalImages)
    print("top-1 accuracy:", top1Accuracy)
    print("top-5 accuracy:", top5Accuracy)
    print("macro precision:", macroPrecision)
    print("macro recall:", macroRecall)
    print("macro F1:", macroF1)
    print("test inference seconds:", inferenceSeconds)

    # Save per-image predictions so the confusion matrix, most-confused pairs,
    # and failure-case images can all be built later without re-running the
    # model. Labels stay as class indices; the two confidence columns are the
    # softmax probability of the top-1 guess and of all five guesses in order.
    with open(PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_index",
            "true_label",
            "predicted_label",
            "top5_labels",
            "top1_confidence",
            "top5_confidences",
        ])
        for i in range(totalImages):
            top5String = " ".join(str(c) for c in top5[i])
            top5ConfidenceString = " ".join(
                format(value, ".6f") for value in top5Confidence[i]
            )
            writer.writerow([
                i,
                labels[i],
                top1[i],
                top5String,
                format(top1Confidence[i], ".6f"),
                top5ConfidenceString,
            ])

    print("predictions saved to:", PREDICTIONS_PATH)

    # Headline numbers in one place, so the report can quote them without
    # re-deriving anything from the CSVs.
    metrics = {
        "checkpoint": os.path.basename(BEST_CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "number_of_test_images": totalImages,
        "number_of_classes": NUM_CLASSES,
        "top1_accuracy": top1Accuracy,
        "top5_accuracy": top5Accuracy,
        "macro_precision": macroPrecision,
        "macro_recall": macroRecall,
        "macro_f1": macroF1,
        "test_inference_seconds": inferenceSeconds,
        "mean_milliseconds_per_image": inferenceSeconds / totalImages * 1000,
        "images_per_second": totalImages / inferenceSeconds,
        "batch_size": BATCH_SIZE,
        "device": str(device),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # One row per class. Sorting this by recall afterwards is the quickest way
    # to find which species the model never learned.
    with open(PER_CLASS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class_index",
            "species_name",
            "precision",
            "recall",
            "f1",
            "support",
            "correct_predictions",
        ])
        for classIndex in classLabels:
            writer.writerow([
                classIndex,
                describeClass(classIndex, speciesNames),
                perClassPrecision[classIndex],
                perClassRecall[classIndex],
                perClassF1[classIndex],
                int(support[classIndex]),
                int(matrix[classIndex, classIndex]),
            ])

    np.savetxt(MATRIX_CSV_PATH, matrix, delimiter=",", fmt="%d")
    saveConfusionMatrix(matrix, MATRIX_IMAGE_PATH)

    # Most-confused class pairs: every off-diagonal cell that is not zero,
    # ranked by how often that mistake was made. np.nonzero avoids walking all
    # 250,000 cells of the matrix in Python.
    confusionRows = []
    trueIndices, predictedIndices = np.nonzero(matrix)
    for trueIndex, predictedIndex in zip(trueIndices, predictedIndices):
        if trueIndex == predictedIndex:
            continue

        confusionRows.append({
            "trueIndex": int(trueIndex),
            "predictedIndex": int(predictedIndex),
            "count": int(matrix[trueIndex, predictedIndex]),
        })

    confusionRows.sort(key=lambda row: row["count"], reverse=True)

    with open(CONFUSIONS_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "true_class_index",
            "true_species_name",
            "predicted_class_index",
            "predicted_species_name",
            "count",
        ])
        for row in confusionRows[:TOP_CONFUSIONS_TO_SAVE]:
            writer.writerow([
                row["trueIndex"],
                describeClass(row["trueIndex"], speciesNames),
                row["predictedIndex"],
                describeClass(row["predictedIndex"], speciesNames),
                row["count"],
            ])

    print("metrics and confusion outputs saved in:", OUTPUT_DIR)

    # Example grids. These need the image file for each test index, which only
    # works if the test loader was built with shuffle=False so that batch order
    # matches dataset order.
    imagePaths = getImagePaths(testDataset, totalImages)

    if imagePaths is None:
        print(
            "could not read image paths from the test dataset, "
            "so the example grids were skipped"
        )
    else:
        correctRecords = []
        incorrectRecords = []

        for i in range(totalImages):
            record = {
                "imagePath": imagePaths[i],
                "trueName": describeClass(labels[i], speciesNames),
                "predictedName": describeClass(top1[i], speciesNames),
                "top1Confidence": top1Confidence[i],
            }

            if top1[i] == labels[i]:
                correctRecords.append(record)
            else:
                incorrectRecords.append(record)

        # Correct examples are sampled at random with a fixed seed, so the same
        # twelve images appear every time the script is run.
        randomGenerator = np.random.default_rng(42)
        if len(correctRecords) > GRID_SIZE:
            selectedPositions = randomGenerator.choice(
                len(correctRecords),
                size=GRID_SIZE,
                replace=False,
            )
            correctExamples = [correctRecords[p] for p in selectedPositions]
        else:
            correctExamples = correctRecords

        # Incorrect examples are the most confident mistakes, which are the
        # informative ones: they show genuinely similar species or label noise
        # rather than images the model was already unsure about.
        incorrectExamples = sorted(
            incorrectRecords,
            key=lambda record: record["top1Confidence"],
            reverse=True,
        )[:GRID_SIZE]

        saveExampleGrid(
            correctExamples,
            CORRECT_GRID_PATH,
            "Correct test predictions",
        )
        saveExampleGrid(
            incorrectExamples,
            INCORRECT_GRID_PATH,
            "High-confidence incorrect test predictions",
        )

        print("example grids saved in:", OUTPUT_DIR)