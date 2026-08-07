import os
import sys

import torch

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_FOLDER)
SCRATCH_FOLDER = os.path.join(PROJECT_ROOT, "Scratch")
sys.path.insert(0, SCRATCH_FOLDER)

from config import NORMALIZE_MEAN, NORMALIZE_STD
from dataset_loading import buildDataLoaders
from device import getDevice
from gradcam import generateHeatmap
from visualize import saveOverlay
from run_pretrained import loadPretrainedModel, loadSpeciesNames

NUM_PAIRS_TO_EXPLAIN = 4
OUTPUT_DIR = os.path.join(SCRIPT_FOLDER, "outputs", "pretrained", "confusable_pairs")


# Forward-only pass over the whole test set. Returns the true and
# predicted class index for every image, in dataset order.
def runFullTestPredictions(model, testLoader, device):
    model.eval()
    trueLabels = []
    predictedLabels = []

    with torch.no_grad():
        for images, labels in testLoader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            trueLabels.extend(labels.tolist())
            predictedLabels.extend(predictions.cpu().tolist())

    return trueLabels, predictedLabels


# Counts each (true, predicted) mistake and ranks them, most common first.
def rankConfusedPairs(trueLabels, predictedLabels):
    counts = {}
    i = 0
    while i < len(trueLabels):
        trueIndex = trueLabels[i]
        predictedIndex = predictedLabels[i]
        if trueIndex != predictedIndex:
            key = (trueIndex, predictedIndex)
            counts[key] = counts.get(key, 0) + 1
        i = i + 1

    pairs = list(counts.items())
    pairs.sort(key=lambda item: item[1], reverse=True)
    return pairs


# Finds the first test image whose true/predicted labels match the pair.
def findExampleIndex(trueLabels, predictedLabels, trueIndex, predictedIndex):
    i = 0
    while i < len(trueLabels):
        if trueLabels[i] == trueIndex and predictedLabels[i] == predictedIndex:
            return i
        i = i + 1
    return None


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = loadPretrainedModel(device)
    speciesNames = loadSpeciesNames()

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=32, numWorkers=0
    )

    print("scanning full test set for predictions...")
    trueLabels, predictedLabels = runFullTestPredictions(model, testLoader, device)

    confusedPairs = rankConfusedPairs(trueLabels, predictedLabels)

    targetLayer = model.layer4

    pairIndex = 0
    while pairIndex < NUM_PAIRS_TO_EXPLAIN:
        trueIndex, predictedIndex = confusedPairs[pairIndex][0]
        count = confusedPairs[pairIndex][1]

        exampleIndex = findExampleIndex(trueLabels, predictedLabels, trueIndex, predictedIndex)
        image, label = testDataset[exampleIndex]
        image = image.unsqueeze(0).to(device)

        heatmap, _ = generateHeatmap(model, targetLayer, image, predictedIndex)

        trueName = speciesNames[trueIndex]
        predictedName = speciesNames[predictedIndex]

        outputPath = os.path.join(OUTPUT_DIR, "pair_" + str(pairIndex) + ".png")
        saveOverlay(image, heatmap, NORMALIZE_MEAN, NORMALIZE_STD, outputPath, 0.4)

        print("pair", pairIndex, ":", trueName, "misclassified as", predictedName, "(", count, "times)")
        print("  saved to:", outputPath)

        pairIndex = pairIndex + 1
