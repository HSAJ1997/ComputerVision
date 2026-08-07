from pathlib import Path
import csv
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import INaturalistSubset


NUM_CLASSES = 500
IMAGE_SIZE = 224
BATCH_SIZE = 32

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "pretrained_finetuned_aug_step_best_adam.pth"
)
TEST_CSV = PROJECT_ROOT / "splits" / "test.csv"
CLASS_FILE = PROJECT_ROOT / "splits" / "selected_classes.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_class_information():
    with CLASS_FILE.open("r", encoding="utf-8") as file:
        class_data = json.load(file)

    classes = sorted(
        class_data["classes"],
        key=lambda item: item["class_index"],
    )

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, but found {len(classes)}."
        )

    return {
        item["class_index"]: {
            "category_id": item["category_id"],
            "species_name": item["species_name"],
        }
        for item in classes
    }


def save_confusion_matrix(matrix, output_path):
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    plt.figure(figsize=(16, 14))
    plt.imshow(normalised, interpolation="nearest", aspect="auto")
    plt.title("Normalised confusion matrix: fine-tuned ResNet-18")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.colorbar(label="Proportion of true class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_example_grid(records, output_path, title, project_root):
    if not records:
        return

    records = records[:12]
    columns = 4
    rows = 3

    figure, axes = plt.subplots(rows, columns, figsize=(14, 11))
    axes = axes.flatten()

    for axis, record in zip(axes, records):
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path

        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))

        axis.set_title(
            f"True: {record['true_species_name']}\n"
            f"Pred: {record['predicted_species_name']}\n"
            f"Confidence: {record['top1_confidence']:.3f}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes[len(records):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)


    required_files = [CHECKPOINT_PATH, TEST_CSV, CLASS_FILE]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cpu")
    class_information = load_class_information()

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = INaturalistSubset(
        csv_path=TEST_CSV,
        project_root=PROJECT_ROOT,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(
        f"Evaluating epoch {checkpoint.get('epoch', 'unknown')} "
        f"on {len(test_dataset)} test images."
    )

    all_labels = []
    all_predictions = []
    prediction_records = []

    top5_correct = 0
    forward_seconds = 0.0
    test_start = time.perf_counter()

    dataset_rows = test_dataset.samples
    row_offset = 0

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(
            test_loader,
            start=1,
        ):
            images = images.to(device)
            labels = labels.to(device)

            forward_start = time.perf_counter()
            outputs = model(images)
            forward_seconds += time.perf_counter() - forward_start

            probabilities = torch.softmax(outputs, dim=1)
            top5_probabilities, top5_indices = probabilities.topk(5, dim=1)
            predictions = top5_indices[:, 0]

            top5_correct += (
                top5_indices == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            label_values = labels.cpu().tolist()
            prediction_values = predictions.cpu().tolist()
            top5_index_values = top5_indices.cpu().tolist()
            top5_probability_values = top5_probabilities.cpu().tolist()

            all_labels.extend(label_values)
            all_predictions.extend(prediction_values)

            for position in range(len(label_values)):
                row = dataset_rows[row_offset + position]
                true_index = label_values[position]
                predicted_index = prediction_values[position]
                five_indices = top5_index_values[position]
                five_probabilities = top5_probability_values[position]

                prediction_records.append({
                    "image_path": row["image_path"],
                    "true_class_index": true_index,
                    "true_category_id": class_information[true_index]["category_id"],
                    "true_species_name": class_information[true_index]["species_name"],
                    "predicted_class_index": predicted_index,
                    "predicted_category_id": class_information[predicted_index]["category_id"],
                    "predicted_species_name": class_information[predicted_index]["species_name"],
                    "correct": true_index == predicted_index,
                    "top1_confidence": five_probabilities[0],
                    "top5_class_indices": "|".join(
                        str(index) for index in five_indices
                    ),
                    "top5_species_names": "|".join(
                        class_information[index]["species_name"]
                        for index in five_indices
                    ),
                    "top5_confidences": "|".join(
                        f"{value:.6f}" for value in five_probabilities
                    ),
                })

            row_offset += len(label_values)

            if batch_number % 100 == 0:
                print(
                    f"Processed batch {batch_number}/{len(test_loader)}"
                )

    total_test_seconds = time.perf_counter() - test_start

    top1_accuracy = accuracy_score(all_labels, all_predictions)
    top5_accuracy = top5_correct / len(all_labels)
    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    class_labels = list(range(NUM_CLASSES))
    per_class_precision, per_class_recall, per_class_f1, support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_labels,
    )

    metrics = {
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_macro_f1_used_for_selection": checkpoint.get(
            "validation_macro_f1"
        ),
        "number_of_test_images": len(all_labels),
        "number_of_classes": NUM_CLASSES,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "overall_accuracy": top1_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "forward_inference_seconds": forward_seconds,
        "end_to_end_test_seconds": total_test_seconds,
        "mean_forward_milliseconds_per_image": (
            forward_seconds / len(all_labels) * 1000
        ),
        "forward_images_per_second": (
            len(all_labels) / forward_seconds
        ),
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "device": str(device),
    }

    metrics_path = OUTPUT_DIR / "pretrained_aug_step_adam_test_metrics.json"
    predictions_path = (
        OUTPUT_DIR / "pretrained_aug_step_adam_test_predictions.csv"
    )
    per_class_path = (
        OUTPUT_DIR / "pretrained_aug_step_adam_per_class_metrics.csv"
    )
    matrix_csv_path = (
        OUTPUT_DIR / "pretrained_aug_step_adam_confusion_matrix.csv"
    )
    matrix_image_path = (
        OUTPUT_DIR / "pretrained_aug_step_adam_confusion_matrix.png"
    )
    confusions_path = (
        OUTPUT_DIR / "pretrained_aug_step_adam_common_confusions.csv"
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=prediction_records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(prediction_records)

    per_class_rows = []
    for class_index in class_labels:
        per_class_rows.append({
            "class_index": class_index,
            "category_id": class_information[class_index]["category_id"],
            "species_name": class_information[class_index]["species_name"],
            "precision": per_class_precision[class_index],
            "recall": per_class_recall[class_index],
            "f1": per_class_f1[class_index],
            "support": int(support[class_index]),
            "correct_predictions": int(matrix[class_index, class_index]),
        })

    with per_class_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=per_class_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    np.savetxt(
        matrix_csv_path,
        matrix,
        delimiter=",",
        fmt="%d",
    )
    save_confusion_matrix(matrix, matrix_image_path)

    confusion_rows = []
    for true_index in class_labels:
        for predicted_index in class_labels:
            if true_index == predicted_index:
                continue

            count = int(matrix[true_index, predicted_index])
            if count == 0:
                continue

            confusion_rows.append({
                "true_class_index": true_index,
                "true_species_name": class_information[true_index]["species_name"],
                "predicted_class_index": predicted_index,
                "predicted_species_name": class_information[predicted_index]["species_name"],
                "count": count,
            })

    confusion_rows.sort(
        key=lambda row: row["count"],
        reverse=True,
    )

    with confusions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "true_class_index",
            "true_species_name",
            "predicted_class_index",
            "predicted_species_name",
            "count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(confusion_rows[:50])

    correct_records = [
        record for record in prediction_records
        if record["correct"]
    ]
    incorrect_records = [
        record for record in prediction_records
        if not record["correct"]
    ]

    random_generator = np.random.default_rng(42)
    if len(correct_records) > 12:
        selected_positions = random_generator.choice(
            len(correct_records),
            size=12,
            replace=False,
        )
        correct_examples = [
            correct_records[position]
            for position in selected_positions
        ]
    else:
        correct_examples = correct_records

    incorrect_examples = sorted(
        incorrect_records,
        key=lambda record: record["top1_confidence"],
        reverse=True,
    )[:12]

    save_example_grid(
        correct_examples,
        OUTPUT_DIR / "pretrained_aug_correct_examples.png",
        "Correct test predictions",
        PROJECT_ROOT,
    )
    save_example_grid(
        incorrect_examples,
        OUTPUT_DIR / "pretrained_aug_incorrect_examples.png",
        "High-confidence incorrect test predictions",
        PROJECT_ROOT,
    )

    print(
        f"Top-1: {top1_accuracy * 100:.2f}% | "
        f"Top-5: {top5_accuracy * 100:.2f}% | "
        f"Macro-F1: {macro_f1:.4f}"
    )
    print(f"Results saved in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()