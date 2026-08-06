#!/usr/bin/env python3
"""Evaluate the fixed pretrained ResNet-18 under test-time image degradations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18

from data.dataset import INaturalistSubset
from robustness.degradations import ImageDegradation, available_degradations
from robustness.results import plot_robustness_results, upsert_result


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Checkpoint produced by the pretrained ResNet-18 training scripts.",
    )
    parser.add_argument(
        "--model-type",
        choices=["pretrained", "scratch"],
        default="pretrained",
        help="Architecture used by the checkpoint.",
    )
    parser.add_argument(
        "--model-label",
        default="pretrained_resnet18_stage2",
        help="Name written to the robustness results CSV and plot legends.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/robustness.json"))
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or mps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Evaluate only the clean test set and skip all degradations.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=f"Optional degradation subset. Available: {available_degradations()}",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_from_root(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_count(selected_classes_path: Path) -> int:
    data = load_json(selected_classes_path)
    classes = data.get("classes", [])
    if not classes:
        raise ValueError(f"No classes found in {selected_classes_path}")
    indices = sorted(int(item["class_index"]) for item in classes)
    if indices != list(range(len(classes))):
        raise ValueError("selected_classes.json must contain continuous class indices.")
    return len(classes)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    expected_classes: int,
    model_type: str,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint does not contain 'model_state_dict'.")

    num_classes = int(
        checkpoint.get("number_of_classes", checkpoint.get("num_classes", expected_classes))
    )
    if num_classes != expected_classes:
        raise ValueError(
            f"Checkpoint has {num_classes} classes, but selected_classes.json has "
            f"{expected_classes}."
        )

    if model_type == "pretrained":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "scratch":
        from models.scratch_resnet18 import buildResnet18

        model = buildResnet18(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def build_eval_transforms(
    image_size: int,
    resize_size: int,
    preprocess_mode: str,
):
    if preprocess_mode == "resize_crop":
        pre_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
            ]
        )
    elif preprocess_mode == "resize_square":
        pre_transform = transforms.Resize((image_size, image_size))
    else:
        raise ValueError(
            "preprocess_mode must be either 'resize_crop' or 'resize_square'."
        )
    post_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return pre_transform, post_transform


def make_dataset(
    *,
    test_csv: Path,
    project_root: Path,
    pre_transform,
    post_transform,
    degradation,
    max_samples: int | None,
):
    dataset = INaturalistSubset(
        csv_path=test_csv,
        project_root=project_root,
        pre_transform=pre_transform,
        degradation=degradation,
        post_transform=post_transform,
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


@torch.inference_mode()
def evaluate_condition(
    *,
    model,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    description: str,
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    top5_correct = 0
    total_samples = 0
    use_amp = amp and device.type == "cuda"
    started = time.perf_counter()

    print(f"\n{description}: {len(dataset)} images")
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

        predictions = logits.argmax(dim=1)
        k = min(5, logits.shape[1])
        topk = logits.topk(k=k, dim=1).indices
        top5_correct += int(topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item())

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        total_samples += labels.size(0)

        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"  batch {batch_index}/{len(loader)}")

    elapsed = time.perf_counter() - started
    top1 = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "top1_accuracy": top1,
        "overall_accuracy": top1,
        "top5_accuracy": float(top5_correct / total_samples),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "num_samples": int(total_samples),
        "inference_seconds": float(elapsed),
        "images_per_second": float(total_samples / elapsed) if elapsed > 0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    project_root = Path(config.get("project_root", ".")).resolve()

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)

    checkpoint_path = resolve_from_root(project_root, args.checkpoint)
    test_csv = resolve_from_root(project_root, config["test_csv"])
    selected_classes_path = resolve_from_root(
        project_root, config["selected_classes_json"]
    )
    results_csv = resolve_from_root(project_root, config["results_csv"])
    figures_dir = resolve_from_root(project_root, config["figures_dir"])

    expected_classes = load_class_count(selected_classes_path)
    model, checkpoint = load_model(
        checkpoint_path, device, expected_classes, args.model_type
    )

    image_size = int(config.get("image_size", checkpoint.get("image_size", 224)))
    resize_size = int(config.get("resize_size", image_size))
    preprocess_mode = str(config.get("preprocess_mode", "resize_crop"))
    pre_transform, post_transform = build_eval_transforms(
        image_size, resize_size, preprocess_mode
    )

    batch_size = args.batch_size or int(config.get("batch_size", 64))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config.get("num_workers", 0))
    )
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else config.get("max_samples")
    )
    max_samples = int(max_samples) if max_samples is not None else None
    amp = bool(config.get("amp", True)) and not args.no_amp

    configured = config.get("degradations", {})
    selected_names = [] if args.clean_only else list(configured)
    if args.only is not None and not args.clean_only:
        unknown = set(args.only) - set(configured)
        if unknown:
            raise ValueError(f"Requested degradations are not configured: {sorted(unknown)}")
        selected_names = args.only

    print("=" * 72)
    print("COMP9517 robustness-to-image-degradation experiment")
    print("=" * 72)
    print(f"Model type:        {args.model_type}")
    print(f"Model label:       {args.model_label}")
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Checkpoint epoch:  {checkpoint.get('epoch', 'unknown')}")
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(device)}")
    if preprocess_mode == "resize_square":
        preprocessing_text = f"Resize(({image_size}, {image_size}))"
    else:
        preprocessing_text = f"Resize({resize_size}) -> CenterCrop({image_size})"
    print(f"Image preprocessing: {preprocessing_text}")
    print(f"Batch size:        {batch_size}")
    print(f"Workers:           {num_workers}")
    print(f"Max samples:       {max_samples or 'all'}")
    print(f"Degradations:      {selected_names}")
    print(f"Results CSV:       {results_csv}")

    if bool(config.get("include_clean", True)):
        clean_dataset = make_dataset(
            test_csv=test_csv,
            project_root=project_root,
            pre_transform=pre_transform,
            post_transform=post_transform,
            degradation=None,
            max_samples=max_samples,
        )
        metrics = evaluate_condition(
            model=model,
            dataset=clean_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
            description="Clean test",
        )
        upsert_result(
            results_csv,
            {
                "model": args.model_label,
                "degradation": "clean",
                "severity_level": 0,
                "parameter": "none",
                **metrics,
            },
        )
        print(
            f"clean: top1={metrics['top1_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    for name in selected_names:
        for severity in configured[name]:
            degradation = ImageDegradation(name=name, severity=int(severity), seed=seed)
            dataset = make_dataset(
                test_csv=test_csv,
                project_root=project_root,
                pre_transform=pre_transform,
                post_transform=post_transform,
                degradation=degradation,
                max_samples=max_samples,
            )
            metrics = evaluate_condition(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                amp=amp,
                description=f"{name} severity {severity}",
            )
            upsert_result(
                results_csv,
                {
                    "model": args.model_label,
                    "degradation": name,
                    "severity_level": int(severity),
                    "parameter": degradation.parameter_text,
                    **metrics,
                },
            )
            print(
                f"{name} severity={severity} ({degradation.parameter_text}): "
                f"top1={metrics['top1_accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

    plot_robustness_results(results_csv, figures_dir)

    run_summary = {
        "model": args.model_label,
        "model_type": args.model_type,
        "checkpoint": str(checkpoint_path),
        "test_csv": str(test_csv),
        "selected_classes_json": str(selected_classes_path),
        "seed": seed,
        "image_size": image_size,
        "resize_size": resize_size,
        "preprocess_mode": preprocess_mode,
        "max_samples": max_samples,
        "degradations": selected_names,
        "results_csv": str(results_csv),
    }
    safe_label = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.model_label
    )
    summary_path = results_csv.with_name(
        f"{results_csv.stem}.{safe_label}.run.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)

    print("\nRobustness experiment complete.")
    print(f"Results: {results_csv}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
