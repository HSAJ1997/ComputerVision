# COMP9517 Species Classification Project

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

--Scratch CNN--

The from-scratch ResNet-18 pipeline (separate from the pretrained baseline in scripts/)
lives in Scratch/, split into small, single-purpose files. It reuses the existing
splits/*.csv and splits/selected_classes.json files as-is, class_index (0-499) is already
a consistent class-to-integer mapping shared across train/validation/test, so no new
mapping is built.

--Scratch/dataset_loading.py--

Defines the PyTorch Dataset and builds the three DataLoaders.

INaturalistDataset is a PyTorch Dataset for one split (train, validation or test).
__init__ reads the split CSV once and keeps (imagePath, classIndex) pairs in memory.
__len__ reports how many samples are in the split. __getitem__(index) loads one image
from disk with PIL, converts it to RGB (some source images are grayscale or have an
alpha channel), applies the transform, and returns (image, classIndex). PyTorch calls
__getitem__ on demand per sample, not all at once, keeping memory usage low across
30,000 images.

buildBasicTransform resizes every image to 224x224 (the standard ResNet input size) and
converts it to a tensor. No normalization or augmentation yet, that comes later.

buildDataLoaders builds all three datasets with the same transform (so every split is
preprocessed identically) and wraps each in a DataLoader: batch_size 32, shuffle True
only for training (shuffling validation/test adds nothing but overhead), num_workers 2
to load images in parallel background processes.

printLoaderInfo and the if __name__ == "__main__" block are a verification step that
pulls one batch per loader (not a full pass) and prints image batch shape, label batch
shape, sample labels, and images per split, to confirm the loaders are wired up
correctly.

--Scratch/device.py--

One function, getDevice(), that checks torch.cuda.is_available() (NVIDIA GPU) first,
then torch.backends.mps.is_available() (Apple Silicon GPU), and falls back to cpu if
neither is available. Every other file calls this once and passes the result around, so
the same code runs unchanged on a Windows/NVIDIA PC and on an Apple Silicon Mac.

--Scratch/model.py--

The ResNet-18 architecture built from scratch, not torchvision.models.resnet18, no
pretrained weights.

BasicBlock is the fundamental repeating unit. Two 3x3 convolutions (with batch norm and
ReLU after the first), then the block's own input is added back onto the output before
the final ReLU, the residual connection that lets gradients skip past the block during
training. If a block changes the channel count or shrinks the image (stride 2), the raw
input can't be added directly since shapes won't match, so self.shortcut is a small 1x1
conv that only exists when needed, purely to reshape the input so the addition works.

makeStage stacks several BasicBlocks that all produce the same number of output
channels. Only the first block in a stage is allowed to change shape (downsample),
every block after that keeps the shape steady.

ResNet18 is the full network: self.stem is a 7x7 conv plus maxpool that aggressively
shrinks a 224x224 image before the expensive residual stages start, then four stages
(stage1 to stage4, channels doubling 64 to 128 to 256 to 512 while spatial size shrinks
on stages 2 to 4, the standard ResNet-18 layout of 2 blocks per stage, 8 blocks total),
then self.pool (AdaptiveAvgPool2d, collapsing whatever spatial size is left down to one
value per channel) and self.fc (a single Linear layer mapping those 512 values to
numClasses raw scores).

buildResnet18(numClasses) is a thin factory function so other files don't need to know
the ResNet18 class name directly.

Verified with a shape check: a batch of 4 dummy [4, 3, 224, 224] images produces a
[4, 500] output, and the model has about 11.4M parameters, the expected count for
ResNet-18.

--Scratch/config.py--

Plain constants only, no functions or logic beyond building two file paths.

NUM_CLASSES is 500, matching the dataset. BATCH_SIZE is 32 and NUM_WORKERS is 2, the
same defaults dataset_loading.py used, centralized here. NUM_EPOCHS, LEARNING_RATE,
MOMENTUM and WEIGHT_DECAY are starting hyperparameters for the SGD optimizer, to be
tuned once real training results come in. CHECKPOINT_DIR and BEST_CHECKPOINT_PATH are
where train.py saves model weights during training, kept inside Scratch/checkpoints/
(gitignored, large binary .pth files).

--Scratch/train.py--

The training loop, tying together dataset_loading, device, model and config.

trainOneEpoch is the core training step, one batch at a time: move the batch onto the
right device, optimizer.zero_grad() to clear leftover gradients from the previous batch
(PyTorch accumulates gradients by default), forward pass through the model,
CrossEntropyLoss compares the raw scores against the true class_index to produce one
number for how wrong the model was, loss.backward() computes how much each weight
contributed to that error, and optimizer.step() nudges every weight to reduce it. The
rest of the function tracks running totals to report average loss and accuracy across
the whole epoch, not per batch, loss.item() times images.size(0) weights each batch's
loss by how many images were in it, since the last batch of an epoch can be smaller
than the rest.

validateOneEpoch is structurally identical, but with model.eval() instead of
model.train() (so certain layers behave correctly on unseen data) and everything
wrapped in torch.no_grad() since no weights are being updated, so there is no need to
track gradients.

The if __name__ == "__main__" block is the real driver: builds the device, the three
DataLoaders, the model, the loss function, and an SGD optimizer (a standard choice for
training ResNets from scratch), then loops over epochs. Each epoch trains, then
validates, then prints both. If validation accuracy improved, it saves the model's
weights (model.state_dict()) to BEST_CHECKPOINT_PATH, so the best performing version is
always kept on disk rather than whatever the last epoch happened to produce. This block
is required here, not just stylistic, on Windows a DataLoader with num_workers greater
than 0 re-imports the whole script per worker process, so any top-level training code
needs this guard to avoid recursively re-running in every worker.

Smoke-tested on a few real batches, not a full epoch, end to end: loss came out to
about 6.2, matching ln(500) which is about 6.21, exactly what an untrained, randomly
initialized model should produce on 500 classes, confirming the data to model to loss
to backward to optimizer pipeline works correctly.
