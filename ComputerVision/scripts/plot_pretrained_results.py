from pathlib import Path
import csv

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FROZEN_HISTORY = OUTPUT_DIR / "pretrained_frozen_history.csv"
FINETUNED_HISTORY = OUTPUT_DIR / "pretrained_finetuned_history.csv"
STAGE2_HISTORY = OUTPUT_DIR / "pretrained_finetuned_stage2_history.csv"


def save_comparison_plot(
    frozen_history,
    finetuned_history,
    column,
    ylabel,
    title,
    filename,
    percentage=False,
):
    frozen_values = frozen_history[column]
    finetuned_values = finetuned_history[column]

    if percentage:
        frozen_values = frozen_values * 100
        finetuned_values = finetuned_values * 100

    plt.figure(figsize=(8, 5))
    plt.plot(
        frozen_history["epoch"],
        frozen_values,
        marker="o",
        label="Frozen backbone",
    )
    plt.plot(
        finetuned_history["epoch"],
        finetuned_values,
        marker="o",
        label="Fine-tuned layer4 + fc",
    )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def main():
    required_files = [
        FROZEN_HISTORY,
        FINETUNED_HISTORY,
        STAGE2_HISTORY,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"History file not found: {path}")

    frozen = pd.read_csv(FROZEN_HISTORY)
    finetuned_stage1 = pd.read_csv(FINETUNED_HISTORY)
    finetuned_stage2 = pd.read_csv(STAGE2_HISTORY)

    finetuned = pd.concat(
        [finetuned_stage1, finetuned_stage2],
        ignore_index=True,
    ).sort_values("epoch")

    save_comparison_plot(
        frozen,
        finetuned,
        "train_loss",
        "Training loss",
        "Training loss by epoch",
        "pretrained_training_loss.png",
    )
    save_comparison_plot(
        frozen,
        finetuned,
        "validation_loss",
        "Validation loss",
        "Validation loss by epoch",
        "pretrained_validation_loss.png",
    )
    save_comparison_plot(
        frozen,
        finetuned,
        "train_accuracy",
        "Training accuracy (%)",
        "Training accuracy by epoch",
        "pretrained_training_accuracy.png",
        percentage=True,
    )
    save_comparison_plot(
        frozen,
        finetuned,
        "validation_accuracy",
        "Validation accuracy (%)",
        "Validation accuracy by epoch",
        "pretrained_validation_accuracy.png",
        percentage=True,
    )
    save_comparison_plot(
        frozen,
        finetuned,
        "validation_macro_f1",
        "Validation macro-F1",
        "Validation macro-F1 by epoch",
        "pretrained_validation_macro_f1.png",
    )

    comparison_rows = [
        {
            "experiment": "Frozen backbone",
            "best_epoch": int(
                frozen.loc[
                    frozen["validation_macro_f1"].idxmax(),
                    "epoch",
                ]
            ),
            "best_validation_accuracy": float(
                frozen["validation_accuracy"].max()
            ),
            "best_validation_macro_f1": float(
                frozen["validation_macro_f1"].max()
            ),
            "training_seconds": float(
                frozen["epoch_seconds"].sum()
            ),
        },
        {
            "experiment": "Fine-tuned layer4 + fc",
            "best_epoch": int(
                finetuned.loc[
                    finetuned["validation_macro_f1"].idxmax(),
                    "epoch",
                ]
            ),
            "best_validation_accuracy": float(
                finetuned["validation_accuracy"].max()
            ),
            "best_validation_macro_f1": float(
                finetuned["validation_macro_f1"].max()
            ),
            "training_seconds": float(
                finetuned["epoch_seconds"].sum()
            ),
        },
    ]

    comparison_path = (
        OUTPUT_DIR / "pretrained_validation_comparison.csv"
    )

    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=comparison_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print("Saved the training curves and validation comparison.")


if __name__ == "__main__":
    main()