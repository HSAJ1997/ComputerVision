import json
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
from model import buildResnet18
from selection import saveCorrectAndIncorrect

NUM_CLASSES = 500
NUM_CORRECT_TO_SAVE = 4
NUM_INCORRECT_TO_SAVE = 4
MAX_IMAGES_TO_SCAN = 500

CHECKPOINT_PATH = os.path.join(SCRATCH_FOLDER, "checkpoints", "resnet18_scratch_baseline_optimizer.pth")
CLASS_FILE = os.path.join(PROJECT_ROOT, "splits", "selected_classes.json")
OUTPUT_DIR = os.path.join(SCRIPT_FOLDER, "outputs", "scratch")


# Maps class_index to species_name, for labelling saved images.
def loadSpeciesNames():
    file = open(CLASS_FILE, "r", encoding="utf-8")
    classData = json.load(file)
    file.close()

    names = {}
    for entry in classData["classes"]:
        names[entry["class_index"]] = entry["species_name"]
    return names


# Builds the from-scratch resnet18 and loads the checkpoint.
def loadScratchModel(device):
    model = buildResnet18(NUM_CLASSES)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    model = loadScratchModel(device)
    model.eval()

    speciesNames = loadSpeciesNames()

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=1, numWorkers=0
    )

    targetLayer = model.stage4

    saveCorrectAndIncorrect(
        model, targetLayer, testLoader, speciesNames, OUTPUT_DIR,
        NUM_CORRECT_TO_SAVE, NUM_INCORRECT_TO_SAVE, MAX_IMAGES_TO_SCAN,
        NORMALIZE_MEAN, NORMALIZE_STD, device,
    )
