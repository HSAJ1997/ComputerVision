import json
import os
import sys

import torch
import torch.nn as nn
from torchvision.models import resnet18

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_FOLDER)
SCRATCH_FOLDER = os.path.join(PROJECT_ROOT, "Scratch")
sys.path.insert(0, SCRATCH_FOLDER)

from config import NORMALIZE_MEAN, NORMALIZE_STD
from dataset_loading import buildDataLoaders
from device import getDevice
from gradcam import generateHeatmap
from visualize import saveOverlay

NUM_CLASSES = 500
NUM_IMAGES_TO_EXPLAIN = 8

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "pretrained_finetuned_best_aug_step.pth")
CLASS_FILE = os.path.join(PROJECT_ROOT, "splits", "selected_classes.json")
OUTPUT_DIR = os.path.join(SCRIPT_FOLDER, "outputs", "pretrained")


# Maps class_index to species_name, for labelling saved images.
def loadSpeciesNames():
    file = open(CLASS_FILE, "r", encoding="utf-8")
    classData = json.load(file)
    file.close()

    names = {}
    for entry in classData["classes"]:
        names[entry["class_index"]] = entry["species_name"]
    return names


# Builds the pretrained resnet18 (500-way head) and loads the checkpoint.
def loadPretrainedModel(device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = loadPretrainedModel(device)
    model.eval()

    speciesNames = loadSpeciesNames()

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=1, numWorkers=0
    )

    targetLayer = model.layer4

    imageIndex = 0
    for images, labels in testLoader:
        if imageIndex >= NUM_IMAGES_TO_EXPLAIN:
            break

        images = images.to(device)
        trueClassIndex = int(labels[0].item())

        heatmap, predictedClassIndex = generateHeatmap(model, targetLayer, images, None)

        trueName = speciesNames[trueClassIndex]
        predictedName = speciesNames[predictedClassIndex]
        correct = predictedClassIndex == trueClassIndex

        outputPath = os.path.join(OUTPUT_DIR, "test_image_" + str(imageIndex) + ".png")
        saveOverlay(images, heatmap, NORMALIZE_MEAN, NORMALIZE_STD, outputPath, 0.4)

        print("image", imageIndex)
        print("  true species:", trueName)
        print("  predicted species:", predictedName)
        print("  correct:", correct)
        print("  saved to:", outputPath)

        imageIndex = imageIndex + 1
