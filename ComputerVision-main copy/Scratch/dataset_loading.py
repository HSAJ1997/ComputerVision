import csv
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


# The transform used for now: resize every image to the same fixed size
# and convert it to a tensor. This is only the minimum needed to load
# a batch. Proper preprocessing/augmentation comes in the next step.
def buildBasicTransform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform


# Builds the three datasets and their DataLoaders using the same
# transform, so every split is preprocessed the same way.
def buildDataLoaders(batchSize=32, numWorkers=2):
    transform = buildBasicTransform()

    trainDataset = INaturalistDataset(trainCsvPath, projectRoot, transform)
    validationDataset = INaturalistDataset(validationCsvPath, projectRoot, transform)
    testDataset = INaturalistDataset(testCsvPath, projectRoot, transform)

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
