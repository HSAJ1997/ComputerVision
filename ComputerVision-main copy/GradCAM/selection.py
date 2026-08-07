import os

import torch

from gradcam import generateHeatmap
from visualize import saveOverlay


# Scans the test set, saving Grad-CAM heatmaps for the first numCorrect
# correctly-classified and numIncorrect misclassified images found.
def saveCorrectAndIncorrect(
    model, targetLayer, testLoader, speciesNames, outputDir,
    numCorrect, numIncorrect, maxScan, mean, std, device,
):
    correctDir = os.path.join(outputDir, "correct")
    incorrectDir = os.path.join(outputDir, "incorrect")
    os.makedirs(correctDir, exist_ok=True)
    os.makedirs(incorrectDir, exist_ok=True)

    correctSaved = 0
    incorrectSaved = 0
    imagesScanned = 0

    for images, labels in testLoader:
        imagesScanned = imagesScanned + 1
        if imagesScanned > maxScan:
            break
        if correctSaved >= numCorrect and incorrectSaved >= numIncorrect:
            break

        images = images.to(device)
        trueClassIndex = int(labels[0].item())

        heatmap, predictedClassIndex = generateHeatmap(model, targetLayer, images, None)
        correct = predictedClassIndex == trueClassIndex

        if correct and correctSaved >= numCorrect:
            continue
        if not correct and incorrectSaved >= numIncorrect:
            continue

        trueName = speciesNames[trueClassIndex]
        predictedName = speciesNames[predictedClassIndex]

        if correct:
            outputPath = os.path.join(correctDir, "correct_" + str(correctSaved) + ".png")
            correctSaved = correctSaved + 1
        else:
            outputPath = os.path.join(incorrectDir, "incorrect_" + str(incorrectSaved) + ".png")
            incorrectSaved = incorrectSaved + 1

        saveOverlay(images, heatmap, mean, std, outputPath, 0.4)

        print("true species:", trueName)
        print("  predicted species:", predictedName)
        print("  correct:", correct)
        print("  saved to:", outputPath)

    print("")
    print("correct examples saved:", correctSaved)
    print("incorrect examples saved:", incorrectSaved)
    print("images scanned:", imagesScanned)


# Forward-only pass over the whole test set, for confusion counting.
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


# Ranks the most common mistakes, then saves a Grad-CAM heatmap for one
# example image of each of the top pairs.
def saveConfusablePairs(
    model, targetLayer, testLoader, testDataset, speciesNames, outputDir,
    numPairs, mean, std, device,
):
    os.makedirs(outputDir, exist_ok=True)

    print("scanning full test set for predictions...")
    trueLabels, predictedLabels = runFullTestPredictions(model, testLoader, device)
    confusedPairs = rankConfusedPairs(trueLabels, predictedLabels)

    pairIndex = 0
    while pairIndex < numPairs:
        trueIndex, predictedIndex = confusedPairs[pairIndex][0]
        count = confusedPairs[pairIndex][1]

        exampleIndex = findExampleIndex(trueLabels, predictedLabels, trueIndex, predictedIndex)
        image, label = testDataset[exampleIndex]
        image = image.unsqueeze(0).to(device)

        heatmap, _ = generateHeatmap(model, targetLayer, image, predictedIndex)

        trueName = speciesNames[trueIndex]
        predictedName = speciesNames[predictedIndex]

        outputPath = os.path.join(outputDir, "pair_" + str(pairIndex) + ".png")
        saveOverlay(image, heatmap, mean, std, outputPath, 0.4)

        print("pair", pairIndex, ":", trueName, "misclassified as", predictedName, "(", count, "times)")
        print("  saved to:", outputPath)

        pairIndex = pairIndex + 1
