import torch
import torch.nn as nn

from config import NUM_CLASSES, BATCH_SIZE, NUM_WORKERS, BEST_CHECKPOINT_PATH
from dataset_loading import buildDataLoaders
from device import getDevice
from model import buildResnet18
from train import validateOneEpoch

if __name__ == "__main__":
    device = getDevice()
    print("using device:", device)

    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders(
        batchSize=BATCH_SIZE, numWorkers=NUM_WORKERS
    )

    model = buildResnet18(NUM_CLASSES)
    model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=device))
    model = model.to(device)

    lossFunction = nn.CrossEntropyLoss()

    testLoss, testAccuracy = validateOneEpoch(model, testLoader, lossFunction, device)

    print("test loss:", testLoss)
    print("test accuracy:", testAccuracy)
