"""Deterministic test-time image degradation operators."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageEnhance, ImageFilter


DEGRADATION_PARAMETERS: dict[str, list[float | int]] = {
    "gaussian_noise": [8, 16, 24, 32, 40],          # pixel standard deviation
    "gaussian_blur": [0.5, 1.0, 1.5, 2.0, 3.0],    # PIL blur radius
    "motion_blur": [3, 5, 7, 9, 13],               # diagonal kernel length
    "jpeg_compression": [80, 60, 40, 20, 10],      # JPEG quality
    # Optional extra corruptions. They are not enabled in the default config.
    "brightness": [0.85, 0.70, 0.55, 0.40, 0.25],
    "contrast": [0.85, 0.70, 0.55, 0.40, 0.25],
}


def available_degradations() -> list[str]:
    return sorted(DEGRADATION_PARAMETERS)


def _stable_seed(base_seed: int, sample_key: str, name: str, severity: int) -> int:
    text = f"{base_seed}|{name}|{severity}|{sample_key}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _motion_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    """Apply a deterministic diagonal motion-blur kernel."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    diagonal = torch.arange(kernel_size)
    kernel[diagonal, diagonal] = 1.0 / kernel_size
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    pad = kernel_size // 2
    padded = functional.pad(tensor, (pad, pad, pad, pad), mode="reflect")
    blurred = functional.conv2d(padded, kernel, groups=3)
    blurred = blurred.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray(np.round(blurred * 255.0).astype(np.uint8), mode="RGB")


@dataclass(frozen=True)
class ImageDegradation:
    """Callable corruption with a severity level in the inclusive range 1-5."""

    name: str
    severity: int
    seed: int = 42

    def __post_init__(self) -> None:
        if self.name not in DEGRADATION_PARAMETERS:
            raise ValueError(
                f"Unknown degradation '{self.name}'. "
                f"Available: {available_degradations()}"
            )
        if self.severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be one of 1, 2, 3, 4, 5.")

    @property
    def parameter(self) -> float | int:
        return DEGRADATION_PARAMETERS[self.name][self.severity - 1]

    @property
    def parameter_text(self) -> str:
        names = {
            "gaussian_noise": "sigma",
            "gaussian_blur": "radius",
            "motion_blur": "kernel_size",
            "jpeg_compression": "quality",
            "brightness": "factor",
            "contrast": "factor",
        }
        return f"{names[self.name]}={self.parameter}"

    def __call__(self, image: Image.Image, sample_key: str = "") -> Image.Image:
        image = image.convert("RGB")
        parameter = self.parameter

        if self.name == "gaussian_noise":
            array = np.asarray(image, dtype=np.float32)
            rng = np.random.default_rng(
                _stable_seed(self.seed, sample_key, self.name, self.severity)
            )
            noise = rng.normal(0.0, float(parameter), size=array.shape)
            result = np.clip(array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(result, mode="RGB")

        if self.name == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(radius=float(parameter)))

        if self.name == "motion_blur":
            return _motion_blur(image, kernel_size=int(parameter))

        if self.name == "jpeg_compression":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(parameter), optimize=False)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return compressed.convert("RGB").copy()

        if self.name == "brightness":
            return ImageEnhance.Brightness(image).enhance(float(parameter))

        if self.name == "contrast":
            return ImageEnhance.Contrast(image).enhance(float(parameter))

        raise AssertionError("Unreachable degradation branch.")
