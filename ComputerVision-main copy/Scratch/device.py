import torch


# Picks the best available device: an NVIDIA GPU (cuda) first, then an
# Apple Silicon GPU (mps), and plain cpu if neither GPU is available.
# Using this everywhere means the same code runs unchanged on a
# Windows/NVIDIA PC and on an Apple Silicon Mac.
def getDevice():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    print(getDevice())
