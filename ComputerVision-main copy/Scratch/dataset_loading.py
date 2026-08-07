import csv
import os

from config import (
    USE_AUGMENTATION,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    IMAGE_SIZE
)

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Folder that contains "splits/" and "subset/" (one level above this script).
scriptFolder = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptFolder)
splitsFolder = os.path.join(projectRoot, "splits")

trainCsvPath = os.path.join(splitsFolder, "train.csv")
validationCsvPath = os.path.join(splitsFolder, "validation.csv")
testCsvPath = os.path.join(splitsFolder, "test.csv")


# A PyTorch Dataset for one split (train, validation or test).
# It reads the split CSV once in __init__ and keeps (imagePath, classIndex)
# pairs in memory. class_index already comes from splits/selected_classes.json
# and is identical across train/validation/test, so it is reused directly
# as the label instead of building a separate mapping.
class INaturalistDataset(Dataset):
    def __init__(self, csvPath, projectRoot, transform):
        self.projectRoot = projectRoot
        self.transform = transform
        self.samples = []

        file = open(csvPath, "r", newline="", encoding="utf-8")
        reader = csv.DictReader(file)
        for row in reader:
            imagePath = row["image_path"]
            classIndex = int(row["class_index"])
            self.samples.append((imagePath, classIndex))
        file.close()

    # Tells PyTorch how many samples are in this split.
    def __len__(self):
        return len(self.samples)

    # Loads one image from disk, applies the transform, and returns it
    # together with its integer class label. This is called automatically
    # by the DataLoader, once per sample per batch.
    def __getitem__(self, index):
        imagePath, classIndex = self.samples[index]
        fullPath = os.path.join(self.projectRoot, imagePath)

        image = Image.open(fullPath)
        image = image.convert("RGB")
        image = self.transform(image)

        return image, classIndex

# Training transform: If useAugmentation is True, it randomly varies each
# image every time it is loaded, so the model sees a slightly different
# version each epoch avoid overfitting.
def buildTrainTransform(useAugmentation=True):
    if useAugmentation:
        transform = transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
    return transform


# Evaluation transform: for validation and test. The whole idea is to be fully deterministic:
# the same image always produces the same tensor, so measured accuracy
# reflects the model, not random cropping/jitter. Uses the same
# normalization as training so the inputs match what the model expects.
def buildEvalTransform():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return transform

# Builds the three datasets and their DataLoaders. The train split uses
# the augmented transform; validation and test use the deterministic one.
def buildDataLoaders(batchSize=32, numWorkers=2, useAugmentation=True):
    trainTransform = buildTrainTransform(useAugmentation)
    evalTransform = buildEvalTransform()

    trainDataset = INaturalistDataset(trainCsvPath, projectRoot, trainTransform)
    validationDataset = INaturalistDataset(validationCsvPath, projectRoot, evalTransform)
    testDataset = INaturalistDataset(testCsvPath, projectRoot, evalTransform)

    trainLoader = DataLoader(
        trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers
    )
    validationLoader = DataLoader(
        validationDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers
    )
    testLoader = DataLoader(
        testDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers
    )

    return trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset


# Pulls one batch from a loader and prints its shape and a few labels,
# so we can visually confirm the loader is wired up correctly.
def printLoaderInfo(name, loader, dataset):
    imageBatch, labelBatch = next(iter(loader))

    print(name + ":")
    print("  images in split:", len(dataset))
    print("  image batch shape:", imageBatch.shape)
    print("  label batch shape:", labelBatch.shape)
    print("  sample labels:", labelBatch[:10].tolist())
    print("")


if __name__ == "__main__":
    trainLoader, validationLoader, testLoader, trainDataset, validationDataset, testDataset = buildDataLoaders()

    classIndexSet = set()
    for imagePath, classIndex in trainDataset.samples:
        classIndexSet.add(classIndex)

    print("number of classes:", len(classIndexSet))
    print("")

    printLoaderInfo("train", trainLoader, trainDataset)
    printLoaderInfo("validation", validationLoader, validationDataset)
    printLoaderInfo("test", testLoader, testDataset)
