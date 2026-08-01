import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

# A dataset class for loading images and labels from a CSV file

class INaturalistSubset(Dataset):
    def __init__(self, csv_path, project_root, transform=None):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            self.samples = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError(f"No samples were found in {self.csv_path}")

        columns = set(self.samples[0].keys())
        if "image_path" not in columns or "class_index" not in columns:
            raise ValueError("CSV is missing required columns")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        image_path = self.project_root / row["image_path"]
        class_index = int(row["class_index"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, class_index