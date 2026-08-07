import os

# Folder that contains "splits/" and "subset/" (one level above Scratch/).
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_FOLDER)

NUM_CLASSES = 500

BATCH_SIZE = 32
NUM_WORKERS = 2

NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
IMAGE_SIZE = 224

CHECKPOINT_DIR = os.path.join(SCRIPT_FOLDER, "checkpoints")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "resnet18_scratch_Adam_optimizer.pth")

###########################################################################################
##################################### DATASET SETTINGS ####################################
###########################################################################################

# For dataset training
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


USE_AUGMENTATION = True 