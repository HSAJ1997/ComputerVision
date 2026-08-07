import torch
import torch.nn as nn


# One residual block used by ResNet-18: two 3x3 convolutions with a
# skip connection added back onto the output. If the input and output
# shapes don't match (different channel count or stride), the skip
# connection is passed through a 1x1 conv first so the shapes line up.
class BasicBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.shortcut = None
        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels),
            )

    # Runs one block: conv, bn, relu, conv, bn, then adds the
    # (possibly reshaped) input back before the final relu.
    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# Builds one stage of the network: a sequence of BasicBlocks that all
# share the same output channel count. Only the first block in a stage
# changes the spatial size (via stride) or channel count; the rest
# keep the shape the same.
def makeStage(inChannels, outChannels, numBlocks, stride):
    blocks = []
    blocks.append(BasicBlock(inChannels, outChannels, stride))

    i = 1
    while i < numBlocks:
        blocks.append(BasicBlock(outChannels, outChannels, 1))
        i = i + 1

    return nn.Sequential(*blocks)


# The full ResNet-18 network: a stem that shrinks the raw image, four
# stages of BasicBlocks that build up features while shrinking the
# spatial size further, then a global average pool and a single
# linear layer that outputs one score per class.
class ResNet18(nn.Module):
    def __init__(self, numClasses):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = makeStage(64, 64, 2, 1)
        self.stage2 = makeStage(64, 128, 2, 2)
        self.stage3 = makeStage(128, 256, 2, 2)
        self.stage4 = makeStage(256, 512, 2, 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, numClasses)

    # Passes a batch of images through the stem, the four stages, the
    # pooling layer, and finally the classifier to get class scores.
    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Creates a ready-to-use ResNet-18 model for the given number of
# output classes.
def buildResnet18(numClasses):
    return ResNet18(numClasses)
