# COMP9517 Species Classification Project

## Environment Setup

Run the following commands from the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Traditional SIFT-BoVW-SVM

The traditional classification pipeline uses SIFT descriptors, a K-means
visual vocabulary, Bag of Visual Words representations and a linear SVM
classifier.

### Required BoVW Files

The following generated files should be placed in the project root:

```text
bovw_kmeans.pkl
bovw_train_features.npy
bovw_train_labels.npy
bovw_validation_features.npy
bovw_validation_labels.npy
bovw_test_features.npy
bovw_test_labels.npy
```

### Train and Evaluate the Linear SVM

Run the following command from the project root:

```powershell
python scripts/svm.py
```

The script performs the following steps:

1. Loads the training, validation and test BoVW features
2. Evaluates several regularisation values on the validation set
3. Selects the value with the highest validation Macro-F1
4. Retrains the classifier using the combined training and validation sets
5. Evaluates the final classifier on the test set

The generated results are written to:

```text
outputs/traditional/
```

Important outputs include:

```text
outputs/traditional/metrics.json
outputs/traditional/validation_results.csv
outputs/traditional/top_confusions.csv
outputs/traditional/svm_model.pkl
```

## Robustness Evaluation

The robustness pipeline evaluates the following models:

- Pretrained ResNet-18
- ResNet-18 trained from scratch
- Traditional SIFT-BoVW-SVM

Four test-time image degradations are evaluated:

```text
gaussian_noise
gaussian_blur
motion_blur
jpeg_compression
```

Each degradation contains five severity levels. Severity level 0 represents
the clean test set.

Image degradations are applied only to the held-out test images. The trained
CNN weights, K-means vocabulary and SVM classifier remain fixed during the
evaluation. No model is retrained using degraded images.

### Required Local Files

The following files are required to reproduce the robustness experiments:

```text
ComputerVision/
├── subset/
│   ├── train_mini/
│   └── val/
│
├── checkpoints/
│   ├── pretrained_finetuned_best_aug_step.pth
│   └── resnet18_scratch_best.pth
│
├── bovw_kmeans.pkl
├── bovw_test_features.npy
└── bovw_test_labels.npy
```

The trained SVM should be available at:

```text
outputs/traditional/svm_model.pkl
```

### Test the Degradation Implementations

Run:

```powershell
python -m unittest tests.test_degradations
```

Expected output:

```text
Ran 4 tests
OK
```

### Pretrained ResNet-18 Robustness

The pretrained model uses direct resizing to `224 × 224`.

#### Quick Test

```powershell
python scripts/run_robustness.py `
  --checkpoint checkpoints/pretrained_finetuned_best_aug_step.pth `
  --model-type pretrained `
  --model-label pretrained_resnet18_aug `
  --config configs/robustness_quick.json
```

#### Full Evaluation

```powershell
python scripts/run_robustness.py `
  --checkpoint checkpoints/pretrained_finetuned_best_aug_step.pth `
  --model-type pretrained `
  --model-label pretrained_resnet18_aug `
  --config configs/robustness.json
```

### Scratch ResNet-18 Robustness

The scratch model uses direct resizing to `224 × 224`.

#### Quick Test

```powershell
python scripts/run_robustness.py `
  --checkpoint checkpoints/resnet18_scratch_best.pth `
  --model-type scratch `
  --model-label scratch_resnet18 `
  --config configs/robustness_scratch_quick.json
```

#### Full Evaluation

```powershell
python scripts/run_robustness.py `
  --checkpoint checkpoints/resnet18_scratch_best.pth `
  --model-type scratch `
  --model-label scratch_resnet18 `
  --config configs/robustness_scratch.json
```

### Traditional Model Robustness

The traditional robustness pipeline follows this process:

```text
Degraded test image
→ SIFT feature extraction
→ Fixed K-means visual vocabulary
→ BoVW representation
→ Fixed linear SVM
→ Prediction
```

The K-means vocabulary and SVM classifier are not retrained.

#### Quick Test

```powershell
python scripts/run_traditional_robustness.py `
  --config configs/robustness_traditional_quick.json
```

#### Full Evaluation

```powershell
python scripts/run_traditional_robustness.py `
  --config configs/robustness_traditional.json
```

### Run a Single Degradation

Use the `--only` argument to evaluate one degradation.

For example, run Gaussian noise on the pretrained model:

```powershell
python scripts/run_robustness.py `
  --checkpoint checkpoints/pretrained_finetuned_best_aug_step.pth `
  --model-type pretrained `
  --model-label pretrained_resnet18_aug `
  --config configs/robustness.json `
  --only gaussian_noise
```

For the traditional model:

```powershell
python scripts/run_traditional_robustness.py `
  --config configs/robustness_traditional.json `
  --only gaussian_noise
```

Supported values are:

```text
gaussian_noise
gaussian_blur
motion_blur
jpeg_compression
```

### Robustness Outputs

The combined robustness results are stored in:

```text
outputs/robustness/robustness_results.csv
```

Generated robustness figures are written to:

```text
outputs/robustness/figures/
```

Quick-test results are written to separate output directories and do not
overwrite the full experiment results.

### Evaluation Metrics

The robustness experiments report:

- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro-F1
- Inference time
- Images processed per second
