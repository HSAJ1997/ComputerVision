from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


REQUIRED_FILES = {
    "x_train": "bovw_train_features.npy",
    "y_train": "bovw_train_labels.npy",
    "x_validation": "bovw_validation_features.npy",
    "y_validation": "bovw_validation_labels.npy",
    "x_test": "bovw_test_features.npy",
    "y_test": "bovw_test_labels.npy",
}


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Train and evaluate a linear SVM on BoVW features."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root,
        help="Directory containing the six BoVW feature/label .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "traditional",
        help="Directory in which generated results will be saved.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[1e-7, 3e-7, 1e-6, 2e-6, 3e-6],
        help="Regularisation strengths compared on the validation set.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    return np.load(path, allow_pickle=False)


def load_dataset(input_dir: Path) -> dict[str, np.ndarray]:
    print(f"Loading BoVW data from: {input_dir.resolve()}")
    data = {
        key: load_array(input_dir / filename)
        for key, filename in REQUIRED_FILES.items()
    }
    validate_dataset(data)
    return data


def validate_dataset(data: dict[str, np.ndarray]) -> None:
    for split in ("train", "validation", "test"):
        features = data[f"x_{split}"]
        labels = data[f"y_{split}"]

        if features.ndim != 2:
            raise ValueError(f"{split} features must be 2-D, got {features.shape}.")
        if labels.ndim != 1:
            raise ValueError(f"{split} labels must be 1-D, got {labels.shape}.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{split} feature/label counts do not match: "
                f"{features.shape[0]} vs {labels.shape[0]}."
            )
        if not np.isfinite(features).all():
            raise ValueError(f"{split} features contain NaN or infinity.")

    dimensions = {
        data["x_train"].shape[1],
        data["x_validation"].shape[1],
        data["x_test"].shape[1],
    }
    if len(dimensions) != 1:
        raise ValueError("Feature dimensions differ between data splits.")

    train_classes = set(np.unique(data["y_train"]).tolist())
    for split in ("validation", "test"):
        missing = set(np.unique(data[f"y_{split}"]).tolist()) - train_classes
        if missing:
            raise ValueError(
                f"{split} contains labels absent from training data: {sorted(missing)}"
            )

    print("\nDataset summary")
    print("-" * 68)
    for split in ("train", "validation", "test"):
        features = data[f"x_{split}"]
        labels = data[f"y_{split}"]
        zero_vectors = int(np.count_nonzero(np.linalg.norm(features, axis=1) == 0))
        print(
            f"{split.capitalize():<12}: {features.shape[0]:>6} samples, "
            f"{features.shape[1]:>4} features, "
            f"{len(np.unique(labels)):>3} classes, "
            f"{zero_vectors} zero vectors"
        )
    print("-" * 68)


def build_model(alpha: float, max_iter: int, tol: float, seed: int) -> SGDClassifier:
    """Build a scalable linear SVM using hinge loss."""
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        shuffle=True,
        random_state=seed,
        n_jobs=-1,
        average=True,
    )


def calculate_metrics(
    model: SGDClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    start = time.perf_counter()
    scores = model.decision_function(features)
    predictions = model.classes_[np.argmax(scores, axis=1)]
    inference_seconds = time.perf_counter() - start

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    top1 = accuracy_score(labels, predictions)
    top5 = top_k_accuracy_score(
        labels,
        scores,
        k=min(5, len(model.classes_)),
        labels=model.classes_,
    )

    metrics = {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "overall_accuracy": float(top1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "inference_time_seconds": float(inference_seconds),
        "milliseconds_per_image": float(1000 * inference_seconds / len(labels)),
    }
    return metrics, predictions, scores


def tune_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alpha_values: list[float],
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    if not alpha_values or any(alpha <= 0 for alpha in alpha_values):
        raise ValueError("All alpha values must be greater than zero.")

    results: list[dict[str, Any]] = []
    print("\nValidation search")
    print("-" * 68)

    for alpha in alpha_values:
        print(f"Training candidate with alpha={alpha:g} ...")
        model = build_model(alpha, max_iter, tol, seed)

        start = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start

        metrics, _, _ = calculate_metrics(model, x_validation, y_validation)
        result = {
            "alpha": float(alpha),
            "training_time_seconds": float(training_seconds),
            "iterations": int(model.n_iter_),
            **metrics,
        }
        results.append(result)

        print(
            f"  macro-F1={metrics['macro_f1']:.4f}, "
            f"top-1={metrics['top1_accuracy']:.4f}, "
            f"top-5={metrics['top5_accuracy']:.4f}, "
            f"training={training_seconds:.1f}s"
        )

    best = max(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["top1_accuracy"],
            -item["alpha"],
        ),
    )
    print("-" * 68)
    print(f"Selected alpha: {best['alpha']:g}")
    return float(best["alpha"]), results


def save_validation_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    with (output_dir / "validation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ordered = sorted(results, key=lambda item: item["alpha"])
    alphas = [item["alpha"] for item in ordered]
    macro_f1 = [item["macro_f1"] for item in ordered]
    top1 = [item["top1_accuracy"] for item in ordered]

    plt.figure(figsize=(8, 5))
    plt.semilogx(alphas, macro_f1, marker="o", label="Validation macro-F1")
    plt.semilogx(alphas, top1, marker="o", label="Validation top-1 accuracy")
    plt.xlabel("Alpha (L2 regularisation)")
    plt.ylabel("Score")
    plt.title("Linear SVM validation performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_alpha_comparison.png", dpi=200)
    plt.close()


def get_ranked_classes(
    model: SGDClassifier,
    scores: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    candidate_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    ranked_indices = np.take_along_axis(candidate_indices, order, axis=1)
    ranked_scores = np.take_along_axis(candidate_scores, order, axis=1)
    return model.classes_[ranked_indices], ranked_scores


def save_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    model: SGDClassifier,
    output_dir: Path,
) -> None:
    top_classes, top_scores = get_ranked_classes(model, scores)

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "predicted_label",
                "correct",
                "top5_labels",
                "top5_scores",
            ]
        )
        for index, (true_label, predicted_label) in enumerate(
            zip(labels, predictions)
        ):
            writer.writerow(
                [
                    index,
                    int(true_label),
                    int(predicted_label),
                    int(true_label == predicted_label),
                    " ".join(map(str, top_classes[index].tolist())),
                    " ".join(f"{value:.6f}" for value in top_scores[index]),
                ]
            )


def save_class_analysis(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: np.ndarray,
    output_dir: Path,
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=classes,
        average=None,
        zero_division=0,
    )

    with (output_dir / "class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class_label", "precision", "recall", "f1", "support"])
        for class_label, p, r, class_f1, count in zip(
            classes, precision, recall, f1, support
        ):
            writer.writerow(
                [int(class_label), float(p), float(r), float(class_f1), int(count)]
            )

    matrix = confusion_matrix(labels, predictions, labels=classes)
    np.save(output_dir / "confusion_matrix.npy", matrix)

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(normalised, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Proportion of true class")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.title("Row-normalised test confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    strongest_pairs: list[tuple[int, int, int]] = []

    for flat_index in np.argsort(off_diagonal.ravel())[::-1]:
        count = int(off_diagonal.ravel()[flat_index])
        if count <= 0 or len(strongest_pairs) >= 30:
            break
        true_index, predicted_index = np.unravel_index(
            flat_index, off_diagonal.shape
        )
        strongest_pairs.append(
            (
                int(classes[true_index]),
                int(classes[predicted_index]),
                count,
            )
        )

    with (output_dir / "top_confusions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "predicted_label", "count"])
        writer.writerows(strongest_pairs)


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main() -> None:
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.input_dir)

    best_alpha, validation_results = tune_alpha(
        data["x_train"],
        data["y_train"],
        data["x_validation"],
        data["y_validation"],
        args.alpha_values,
        args.max_iter,
        args.tol,
        args.seed,
    )
    save_validation_results(validation_results, args.output_dir)

    print("\nRetraining selected model on train + validation data ...")
    x_final_train = np.concatenate(
        [data["x_train"], data["x_validation"]], axis=0
    )
    y_final_train = np.concatenate(
        [data["y_train"], data["y_validation"]], axis=0
    )

    final_model = build_model(best_alpha, args.max_iter, args.tol, args.seed)
    start = time.perf_counter()
    final_model.fit(x_final_train, y_final_train)
    final_training_seconds = time.perf_counter() - start

    print("Evaluating once on the held-out test set ...")
    test_metrics, predictions, scores = calculate_metrics(
        final_model, data["x_test"], data["y_test"]
    )
    test_metrics["training_time_seconds"] = float(final_training_seconds)
    test_metrics["iterations"] = int(final_model.n_iter_)

    save_predictions(
        data["y_test"], predictions, scores, final_model, args.output_dir
    )
    save_class_analysis(
        data["y_test"], predictions, final_model.classes_, args.output_dir
    )
    joblib.dump(final_model, args.output_dir / "svm_model.pkl")

    summary = {
        "classifier": "Linear SVM trained with SGDClassifier(loss='hinge')",
        "selected_alpha": best_alpha,
        "random_seed": args.seed,
        "max_iter": args.max_iter,
        "tolerance": args.tol,
        "data": {
            "train_samples": int(data["x_train"].shape[0]),
            "validation_samples": int(data["x_validation"].shape[0]),
            "test_samples": int(data["x_test"].shape[0]),
            "feature_dimension": int(data["x_train"].shape[1]),
            "number_of_classes": int(len(np.unique(data["y_train"]))),
        },
        "validation_search": validation_results,
        "test_metrics": test_metrics,
    }
    save_json(summary, args.output_dir / "metrics.json")

    print("\nFinal test results")
    print("-" * 68)
    print(f"Top-1 accuracy : {test_metrics['top1_accuracy']:.4f}")
    print(f"Top-5 accuracy : {test_metrics['top5_accuracy']:.4f}")
    print(f"Macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro recall   : {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1       : {test_metrics['macro_f1']:.4f}")
    print(f"Training time  : {final_training_seconds:.1f}s")
    print(f"Test time      : {test_metrics['inference_time_seconds']:.3f}s")
    print("-" * 68)
    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
