import torch

# Shared storage the hook callbacks write into.
savedActivations = []
savedGradients = []


# Hook: stores the target layer's forward output.
def onForward(module, input, output):
    savedActivations.append(output.detach())


# Hook: stores the gradient flowing into the target layer's output.
def onBackward(module, gradInput, gradOutput):
    savedGradients.append(gradOutput[0].detach())


# Builds a Grad-CAM heatmap for one image and one class (defaults to
# the model's own top prediction if classIndex is None).
def generateHeatmap(model, targetLayer, inputTensor, classIndex):
    savedActivations.clear()
    savedGradients.clear()

    forwardHandle = targetLayer.register_forward_hook(onForward)
    backwardHandle = targetLayer.register_full_backward_hook(onBackward)

    model.zero_grad()
    output = model(inputTensor)

    if classIndex is None:
        classIndex = int(torch.argmax(output, dim=1).item())

    score = output[0, classIndex]
    score.backward()

    forwardHandle.remove()
    backwardHandle.remove()

    activations = savedActivations[0][0]
    gradients = savedGradients[0][0]

    # One importance weight per channel.
    channelWeights = gradients.mean(dim=(1, 2))

    # Weighted sum of the channels into a single heatmap.
    heatmap = torch.zeros(activations.shape[1], activations.shape[2])
    channel = 0
    while channel < activations.shape[0]:
        heatmap = heatmap + channelWeights[channel] * activations[channel]
        channel = channel + 1

    # Only positive influence counts.
    heatmap = torch.relu(heatmap)

    heatmapMax = torch.max(heatmap)
    if heatmapMax > 0:
        heatmap = heatmap / heatmapMax

    return heatmap, classIndex
