import os

# Folder that contains "splits/" and "subset/" (one level above Scratch/).
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_FOLDER)

NUM_CLASSES = 500

BATCH_SIZE = 32
NUM_WORKERS = 4

NUM_EPOCHS = 40
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

CHECKPOINT_DIR = os.path.join(SCRIPT_FOLDER, "checkpoints")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_scratch_best.pth")

USE_AUGMENTATION = False 