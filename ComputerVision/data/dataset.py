import csv
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class INaturalistSubset(Dataset):

    def __init__(
        self,
        csv_path,
        project_root,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        degradation: Optional[Callable] = None,
        post_transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.transform = transform
        self.pre_transform = pre_transform
        self.degradation = degradation
        self.post_transform = post_transform
        self.return_path = return_path

        if transform is not None and any(
            item is not None for item in (pre_transform, degradation, post_transform)
        ):
            raise ValueError(
                "Use either transform, or pre_transform/degradation/post_transform, not both."
            )

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            self.samples = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError(f"No samples were found in {self.csv_path}")

        columns = set(self.samples[0].keys())
        if "image_path" not in columns or "class_index" not in columns:
            raise ValueError("CSV is missing required columns: image_path, class_index")

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, raw_path: str) -> Path:
        image_path = Path(raw_path)
        return image_path if image_path.is_absolute() else self.project_root / image_path

    def __getitem__(self, index):
        row = self.samples[index]
        image_path = self._resolve_image_path(row["image_path"])
        class_index = int(row["class_index"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not read image: {image_path}") from error

        if self.transform is not None:
            image = self.transform(image)
        else:
            if self.pre_transform is not None:
                image = self.pre_transform(image)
            if self.degradation is not None:
                image = self.degradation(image, sample_key=str(image_path))
            if self.post_transform is not None:
                image = self.post_transform(image)

        if self.return_path:
            return image, class_index, str(image_path)
        return image, class_index
