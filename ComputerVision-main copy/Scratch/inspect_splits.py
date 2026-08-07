import csv
import json
import os

# Find the project root (the folder that contains "splits/"),
# which is one level up from this script's folder.
scriptFolder = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptFolder)
splitsFolder = os.path.join(projectRoot, "splits")

classesFile = os.path.join(splitsFolder, "selected_classes.json")
trainFile = os.path.join(splitsFolder, "train.csv")
validationFile = os.path.join(splitsFolder, "validation.csv")
testFile = os.path.join(splitsFolder, "test.csv")


# Read selected_classes.json and return the metadata plus a set of class indices.
def readClassesFile(path):
    file = open(path, "r", encoding="utf-8")
    data = json.load(file)
    file.close()

    classIndexSet = set()
    classIndexToName = {}
    classList = data["classes"]
    for entry in classList:
        index = entry["class_index"]
        classIndexSet.add(index)
        classIndexToName[index] = entry["species_name"]

    return data, classIndexSet, classIndexToName


# Read one split CSV and return the list of rows (each row is a dict).
def readSplitFile(path):
    file = open(path, "r", newline="", encoding="utf-8")
    reader = csv.DictReader(file)
    rows = list(reader)
    file.close()
    return rows


# Print row count, class stats and one sample row for a split.
def printSplitSummary(name, rows):
    rowCount = len(rows)

    classIndexSet = set()
    speciesNameSet = set()
    for row in rows:
        classIndexSet.add(int(row["class_index"]))
        speciesNameSet.add(row["species_name"])

    minClassIndex = min(classIndexSet)
    maxClassIndex = max(classIndexSet)

    print(name + ":")
    print("  rows:", rowCount)
    print("  distinct classes:", len(classIndexSet))
    print("  class_index range:", minClassIndex, "to", maxClassIndex)
    print("  distinct species names:", len(speciesNameSet))
    print("  sample row:", rows[0])
    print("")

    return classIndexSet


# Check that class indices are exactly 0..499 with no gaps or duplicates.
def checkContiguous(classIndexSet, expectedCount):
    isContiguous = True
    i = 0
    while i < expectedCount:
        if i not in classIndexSet:
            isContiguous = False
            break
        i = i + 1

    sameSize = len(classIndexSet) == expectedCount
    return isContiguous and sameSize


# Check whether any image_path values are shared between two splits.
def checkPathOverlap(rowsA, rowsB):
    pathsA = set()
    for row in rowsA:
        pathsA.add(row["image_path"])

    overlapCount = 0
    for row in rowsB:
        if row["image_path"] in pathsA:
            overlapCount = overlapCount + 1

    return overlapCount


print("=== selected_classes.json ===")
classesData, allClassIndices, classIndexToName = readClassesFile(classesFile)
print("seed:", classesData["seed"])
print("number_of_classes:", classesData["number_of_classes"])
print("train_per_class:", classesData["train_per_class"])
print("validation_per_class:", classesData["validation_per_class"])
print("test_per_class:", classesData["test_per_class"])
print("")

print("=== split files ===")
trainRows = readSplitFile(trainFile)
validationRows = readSplitFile(validationFile)
testRows = readSplitFile(testFile)

trainClassIndices = printSplitSummary("train.csv", trainRows)
validationClassIndices = printSplitSummary("validation.csv", validationRows)
testClassIndices = printSplitSummary("test.csv", testRows)

print("=== cross-checks ===")

expectedCount = classesData["number_of_classes"]

contiguousOk = checkContiguous(allClassIndices, expectedCount)
print("selected_classes.json class_index is 0..N-1 with no gaps:", contiguousOk)

trainMatches = trainClassIndices == allClassIndices
validationMatches = validationClassIndices == allClassIndices
testMatches = testClassIndices == allClassIndices
print("train.csv uses exactly the classes in selected_classes.json:", trainMatches)
print("validation.csv uses exactly the classes in selected_classes.json:", validationMatches)
print("test.csv uses exactly the classes in selected_classes.json:", testMatches)

trainValidationOverlap = checkPathOverlap(trainRows, validationRows)
trainTestOverlap = checkPathOverlap(trainRows, testRows)
validationTestOverlap = checkPathOverlap(validationRows, testRows)
print("image_path overlap between train and validation:", trainValidationOverlap)
print("image_path overlap between train and test:", trainTestOverlap)
print("image_path overlap between validation and test:", validationTestOverlap)
