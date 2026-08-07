# Traditional SIFT + BoVW + SVM robustness

The experiment keeps the fitted K-means vocabulary and linear SVM fixed. Only
held-out test images are degraded. Each degraded image is re-encoded from the
image itself; clean `bovw_test_features.npy` is not reused for corrupted tests.

Required local files:

- `bovw_kmeans.pkl`
- `bovw_test_features.npy`
- `bovw_test_labels.npy`
- `outputs/traditional/svm_model.pkl`
- `splits/test.csv`
- `subset/val/...`

Quick check:

```powershell
python scripts/run_traditional_robustness.py --config configs/robustness_traditional_quick.json
```

Full experiment, one degradation at a time (recommended):

```powershell
python scripts/run_traditional_robustness.py --config configs/robustness_traditional.json --only gaussian_noise
python scripts/run_traditional_robustness.py --config configs/robustness_traditional.json --only gaussian_blur
python scripts/run_traditional_robustness.py --config configs/robustness_traditional.json --only motion_blur
python scripts/run_traditional_robustness.py --config configs/robustness_traditional.json --only jpeg_compression
```

Completed conditions are skipped automatically. Add `--overwrite-existing` only
to deliberately replace existing results.
