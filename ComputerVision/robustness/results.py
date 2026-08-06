"""Robustness result persistence and plotting."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_FIELDS = [
    "model",
    "degradation",
    "severity_level",
    "parameter",
    "top1_accuracy",
    "overall_accuracy",
    "top5_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "num_samples",
    "inference_seconds",
    "images_per_second",
]


def upsert_result(csv_path: str | Path, new_row: dict) -> None:
    """Insert or replace a result identified by model/degradation/severity."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))

    key = (
        str(new_row["model"]),
        str(new_row["degradation"]),
        str(new_row["severity_level"]),
    )

    filtered = [
        row
        for row in rows
        if (
            row.get("model"),
            row.get("degradation"),
            row.get("severity_level"),
        )
        != key
    ]
    filtered.append({field: new_row.get(field, "") for field in RESULT_FIELDS})
    filtered.sort(
        key=lambda row: (
            row.get("model", ""),
            row.get("degradation", ""),
            int(row.get("severity_level", 0)),
        )
    )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(filtered)


def read_results(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def plot_robustness_results(csv_path: str | Path, figures_dir: str | Path) -> None:
    """Create top-1 and macro-F1 severity curves for each degradation."""
    rows = read_results(csv_path)
    if not rows:
        return

    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    degradations = sorted(
        {row["degradation"] for row in rows if row["degradation"] != "clean"}
    )
    models = sorted({row["model"] for row in rows})
    clean_by_model = {
        row["model"]: row for row in rows if row["degradation"] == "clean"
    }

    for degradation in degradations:
        degraded_rows = [row for row in rows if row["degradation"] == degradation]

        for metric, ylabel in [
            ("top1_accuracy", "Top-1 accuracy"),
            ("macro_f1", "Macro F1"),
        ]:
            plt.figure(figsize=(8, 5))
            plotted = False

            for model in models:
                model_rows = [row for row in degraded_rows if row["model"] == model]
                model_rows.sort(key=lambda row: int(row["severity_level"]))
                if not model_rows:
                    continue

                x_values: list[int] = []
                y_values: list[float] = []
                clean = clean_by_model.get(model)
                if clean is not None and clean.get(metric, "") != "":
                    x_values.append(0)
                    y_values.append(float(clean[metric]))

                x_values.extend(int(row["severity_level"]) for row in model_rows)
                y_values.extend(float(row[metric]) for row in model_rows)
                plt.plot(x_values, y_values, marker="o", label=model)
                plotted = True

            if not plotted:
                plt.close()
                continue

            plt.xlabel("Severity level (0 = clean)")
            plt.ylabel(ylabel)
            plt.title(f"{degradation.replace('_', ' ').title()}: {ylabel}")
            plt.xticks([0, 1, 2, 3, 4, 5])
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            output = figures / f"robustness_{degradation}_{metric}.png"
            plt.savefig(output, dpi=180)
            plt.close()
