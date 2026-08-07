import os
import sys

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_FOLDER)
SCRATCH_FOLDER = os.path.join(PROJECT_ROOT, "Scratch")
sys.path.insert(0, SCRATCH_FOLDER)

from config import NORMALIZE_MEAN, NORMALIZE_STD
from dataset_loading import buildDataLoaders
from device import getDevice
from selection import saveConfusablePairs
from run_scratch import loadScratchModel, loadSpeciesNames

NUM_PAIRS_TO_EXPLAIN = 4
OUTPUT_DIR = os.path.join(SCRIPT_FOLDER, "outputs", "scratch", "confusable_pairs")

if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    model = loadScratchModel(device)
    model.eval()
    speciesNames = loadSpeciesNames()

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=32, numWorkers=0
    )

    targetLayer = model.stage4

    saveConfusablePairs(
        model, targetLayer, testLoader, testDataset, speciesNames, OUTPUT_DIR,
        NUM_PAIRS_TO_EXPLAIN, NORMALIZE_MEAN, NORMALIZE_STD, device,
    )
