import torch


# Picks cuda, then mps, then falls back to cpu.
def getDevice():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    print(getDevice())
