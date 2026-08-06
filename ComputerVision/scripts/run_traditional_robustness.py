"""Evaluate the fixed SIFT + BoVW + linear-SVM pipeline under test-time degradations.

The K-means visual vocabulary and SVM classifier are never retrained. For every
corrupted condition, the script degrades each held-out test image, extracts SIFT
features, encodes them with the fixed vocabulary, and predicts with the fixed SVM.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from robustness.degradations import ImageDegradation
from robustness.results import plot_robustness_results, read_results, upsert_result

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness experiments for fixed SIFT + BoVW + SVM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT_FROM_SCRIPT / "configs" / "robustness_traditional.json",
    )
    parser.add_argument(
        "--model-label",
        default="traditional_sift_bovw_svm",
        help="Label stored in the shared robustness results CSV.",
    )
    parser.add_argument(
        "--kmeans",
        type=Path,
        default=None,
        help="Override the K-means vocabulary path from the config.",
    )
    parser.add_argument(
        "--svm-model",
        type=Path,
        default=None,
        help="Override the trained SVM model path from the config.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only the named configured degradations.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of test images for a quick check.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Evaluate only the clean cached BoVW test features.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Re-run conditions already present in the results CSV.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many images.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_from_root(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def load_test_rows(csv_path: Path, max_samples: int | None) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No test samples found in {csv_path}")
    required = {"image_path", "class_index"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Test CSV is missing columns: {sorted(missing)}")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        rows = rows[: min(max_samples, len(rows))]
    return rows


def existing_keys(results_csv: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in read_results(results_csv):
        try:
            severity = int(row.get("severity_level", "0"))
        except ValueError:
            continue
        keys.add((row.get("model", ""), row.get("degradation", ""), severity))
    return keys


def load_models(kmeans_path: Path, svm_path: Path):
    if not kmeans_path.is_file():
        raise FileNotFoundError(f"BoVW K-means vocabulary not found: {kmeans_path}")
    if not svm_path.is_file():
        raise FileNotFoundError(f"Trained SVM model not found: {svm_path}")

    print("Loading fixed BoVW vocabulary ...")
    kmeans = joblib.load(kmeans_path)
    print("Loading fixed SVM classifier ...")
    svm = joblib.load(svm_path)

    if not hasattr(kmeans, "predict") or not hasattr(kmeans, "cluster_centers_"):
        raise TypeError("The supplied K-means file is not a fitted clustering model.")
    if not hasattr(svm, "decision_function") or not hasattr(svm, "classes_"):
        raise TypeError("The supplied SVM file is not a fitted classifier.")

    vocabulary_size = int(kmeans.cluster_centers_.shape[0])
    descriptor_dimension = int(kmeans.cluster_centers_.shape[1])
    svm_feature_dimension = int(getattr(svm, "n_features_in_", vocabulary_size))
    if descriptor_dimension != 128:
        raise ValueError(
            f"Expected 128-dimensional SIFT descriptors, got {descriptor_dimension}."
        )
    if svm_feature_dimension != vocabulary_size:
        raise ValueError(
            "SVM feature dimension does not match the BoVW vocabulary: "
            f"{svm_feature_dimension} vs {vocabulary_size}."
        )
    return kmeans, svm, vocabulary_size


def encode_pil_image(
    image: Image.Image,
    *,
    sift,
    kmeans,
    vocabulary_size: int,
) -> np.ndarray:
    """Convert a PIL image to the same L2-normalised BoVW representation."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or descriptors.shape[0] == 0:
        return np.zeros(vocabulary_size, dtype=np.float32)

    word_ids = kmeans.predict(descriptors.astype(np.float32, copy=False))
    histogram = np.bincount(word_ids, minlength=vocabulary_size).astype(np.float32)
    norm = float(np.linalg.norm(histogram))
    if norm > 0:
        histogram /= norm
    return histogram


def calculate_metrics(
    svm,
    features: np.ndarray,
    labels: np.ndarray,
    started_at: float,
) -> dict:
    scores = svm.decision_function(features)
    if scores.ndim != 2:
        raise ValueError(f"Expected multi-class decision scores, got shape {scores.shape}.")
    predictions = svm.classes_[np.argmax(scores, axis=1)]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    top1 = float(accuracy_score(labels, predictions))
    k = min(5, scores.shape[1])
    top_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_classes = svm.classes_[top_indices]
    top5 = float(np.mean(np.any(top_classes == labels[:, None], axis=1)))

    elapsed = time.perf_counter() - started_at

    return {
        "top1_accuracy": top1,
        "overall_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_samples": int(len(labels)),
        "inference_seconds": float(elapsed),
        "images_per_second": float(len(labels) / elapsed) if elapsed > 0 else 0.0,
    }


def evaluate_cached_clean(
    *,
    svm,
    features_path: Path,
    labels_path: Path,
    expected_rows: list[dict[str, str]],
) -> dict:
    if not features_path.is_file() or not labels_path.is_file():
        missing = [str(path) for path in (features_path, labels_path) if not path.is_file()]
        raise FileNotFoundError(
            "Cached clean BoVW files are required for the clean baseline. Missing: "
            + ", ".join(missing)
        )

    features = np.load(features_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    sample_count = len(expected_rows)
    features = features[:sample_count]
    labels = labels[:sample_count]

    expected_labels = np.asarray(
        [int(row["class_index"]) for row in expected_rows], dtype=labels.dtype
    )
    if features.ndim != 2 or len(features) != sample_count:
        raise ValueError("Cached clean features do not match the requested test subset.")
    if labels.ndim != 1 or not np.array_equal(labels, expected_labels):
        raise ValueError("Cached clean labels do not match splits/test.csv order.")

    started = time.perf_counter()
    metrics = calculate_metrics(svm, features, labels, started)
    return metrics


def evaluate_degraded_condition(
    *,
    rows: list[dict[str, str]],
    project_root: Path,
    degradation: ImageDegradation,
    kmeans,
    svm,
    vocabulary_size: int,
    progress_every: int,
) -> tuple[dict, int]:
    features = np.empty((len(rows), vocabulary_size), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    zero_vectors = 0
    sift = cv2.SIFT_create()
    started = time.perf_counter()

    for index, row in enumerate(rows):
        relative_path = row["image_path"]
        image_path = project_root / Path(relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image not found: {image_path}")

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                degraded = degradation(image, sample_key=relative_path)
        except Exception as error:
            raise RuntimeError(f"Failed to read/degrade image: {image_path}") from error

        histogram = encode_pil_image(
            degraded,
            sift=sift,
            kmeans=kmeans,
            vocabulary_size=vocabulary_size,
        )
        features[index] = histogram
        labels[index] = int(row["class_index"])
        if not np.any(histogram):
            zero_vectors += 1

        completed = index + 1
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == len(rows)
        ):
            print(f"  image {completed}/{len(rows)} (zero vectors: {zero_vectors})")

    metrics = calculate_metrics(svm, features, labels, started)
    return metrics, zero_vectors


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    project_root = Path(config.get("project_root", ".")).resolve()

    test_csv = resolve_from_root(project_root, config["test_csv"])
    kmeans_path = resolve_from_root(
        project_root, args.kmeans or config["kmeans_path"]
    )
    svm_path = resolve_from_root(
        project_root, args.svm_model or config["svm_model_path"]
    )
    clean_features_path = resolve_from_root(
        project_root, config["clean_features_path"]
    )
    clean_labels_path = resolve_from_root(project_root, config["clean_labels_path"])
    results_csv = resolve_from_root(project_root, config["results_csv"])
    figures_dir = resolve_from_root(project_root, config["figures_dir"])

    max_samples = args.max_samples
    if max_samples is None and config.get("max_samples") is not None:
        max_samples = int(config["max_samples"])
    rows = load_test_rows(test_csv, max_samples)
    kmeans, svm, vocabulary_size = load_models(kmeans_path, svm_path)

    configured = config.get("degradations", {})
    selected_names = list(configured)
    if args.only is not None:
        unknown = set(args.only) - set(configured)
        if unknown:
            raise ValueError(f"Requested degradations are not configured: {sorted(unknown)}")
        selected_names = args.only

    print("=" * 76)
    print("COMP9517 traditional-model robustness experiment")
    print("=" * 76)
    print(f"Model label:       {args.model_label}")
    print(f"Test CSV:          {test_csv}")
    print(f"K-means:           {kmeans_path}")
    print(f"SVM:               {svm_path}")
    print(f"Vocabulary size:   {vocabulary_size}")
    print(f"Test images:       {len(rows)}")
    print(f"Degradations:      {selected_names}")
    print(f"Results CSV:       {results_csv}")
    print("Execution:         CPU (SIFT/BoVW/classical SVM)")

    completed_keys = existing_keys(results_csv)
    include_clean = bool(config.get("include_clean", True))
    if include_clean:
        clean_key = (args.model_label, "clean", 0)
        if clean_key in completed_keys and not args.overwrite_existing:
            print("\nSkipping clean: result already exists.")
        else:
            print(f"\nClean cached BoVW test: {len(rows)} images")
            metrics = evaluate_cached_clean(
                svm=svm,
                features_path=clean_features_path,
                labels_path=clean_labels_path,
                expected_rows=rows,
            )
            upsert_result(
                results_csv,
                {
                    "model": args.model_label,
                    "degradation": "clean",
                    "severity_level": 0,
                    "parameter": "cached_clean_bovw",
                    **metrics,
                },
            )
            print(
                f"clean: top1={metrics['top1_accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

    if args.clean_only:
        plot_robustness_results(results_csv, figures_dir)
        print("\nClean-only evaluation complete.")
        return

    seed = int(config.get("seed", 42))
    for name in selected_names:
        for severity_value in configured[name]:
            severity = int(severity_value)
            key = (args.model_label, name, severity)
            if key in completed_keys and not args.overwrite_existing:
                print(f"\nSkipping {name} severity {severity}: result already exists.")
                continue

            degradation = ImageDegradation(name=name, severity=severity, seed=seed)
            print(
                f"\n{name} severity {severity} ({degradation.parameter_text}): "
                f"{len(rows)} images"
            )
            metrics, zero_vectors = evaluate_degraded_condition(
                rows=rows,
                project_root=project_root,
                degradation=degradation,
                kmeans=kmeans,
                svm=svm,
                vocabulary_size=vocabulary_size,
                progress_every=args.progress_every,
            )
            upsert_result(
                results_csv,
                {
                    "model": args.model_label,
                    "degradation": name,
                    "severity_level": severity,
                    "parameter": degradation.parameter_text,
                    **metrics,
                },
            )
            completed_keys.add(key)
            print(
                f"{name} severity={severity}: "
                f"top1={metrics['top1_accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"zero_vectors={zero_vectors}"
            )
            plot_robustness_results(results_csv, figures_dir)

    plot_robustness_results(results_csv, figures_dir)
    print("\nTraditional robustness experiment complete.")
    print(f"Results: {results_csv}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
