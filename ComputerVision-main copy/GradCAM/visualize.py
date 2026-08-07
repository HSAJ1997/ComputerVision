import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib import colormaps
from PIL import Image


# Undoes mean/std normalisation, back to a viewable 0-255 image.
def denormalizeImage(inputTensor, mean, std):
    image = inputTensor[0].clone()

    channel = 0
    while channel < 3:
        image[channel] = image[channel] * std[channel] + mean[channel]
        channel = channel + 1

    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


# Upsamples the heatmap to image size and colours it with "jet".
def colorizeHeatmap(heatmap, imageSize):
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = functional.interpolate(
        heatmap, size=(imageSize, imageSize), mode="bilinear", align_corners=False
    )
    heatmap = heatmap.squeeze().cpu().numpy()

    colorMap = colormaps["jet"]
    coloredHeatmap = colorMap(heatmap)[:, :, :3]
    coloredHeatmap = (coloredHeatmap * 255).astype(np.uint8)
    return coloredHeatmap


# Blends the heatmap over the image and saves it as a PNG.
def saveOverlay(inputTensor, heatmap, mean, std, outputPath, alpha):
    originalImage = denormalizeImage(inputTensor, mean, std)
    imageSize = originalImage.shape[0]
    coloredHeatmap = colorizeHeatmap(heatmap, imageSize)

    blended = originalImage.astype(np.float32) * (1 - alpha) + coloredHeatmap.astype(np.float32) * alpha
    blended = blended.astype(np.uint8)

    image = Image.fromarray(blended)
    image.save(outputPath)
