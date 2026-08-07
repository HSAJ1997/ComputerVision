"""Basic tests for image degradation operators."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from robustness.degradations import ImageDegradation


class DegradationTests(unittest.TestCase):
    def setUp(self) -> None:
        array = np.zeros((64, 64, 3), dtype=np.uint8)
        array[:, :32, 0] = 255
        array[:, 32:, 1] = 255
        self.image = Image.fromarray(array, mode="RGB")

    def test_required_degradations_preserve_size_and_mode(self) -> None:
        for name in [
            "gaussian_noise",
            "gaussian_blur",
            "motion_blur",
            "jpeg_compression",
        ]:
            with self.subTest(name=name):
                output = ImageDegradation(name, severity=3)(self.image, "sample.jpg")
                self.assertEqual(output.size, self.image.size)
                self.assertEqual(output.mode, "RGB")

    def test_noise_is_deterministic_for_same_sample(self) -> None:
        degradation = ImageDegradation("gaussian_noise", severity=2, seed=42)
        first = np.asarray(degradation(self.image, "same.jpg"))
        second = np.asarray(degradation(self.image, "same.jpg"))
        self.assertTrue(np.array_equal(first, second))

    def test_noise_differs_for_different_samples(self) -> None:
        degradation = ImageDegradation("gaussian_noise", severity=2, seed=42)
        first = np.asarray(degradation(self.image, "a.jpg"))
        second = np.asarray(degradation(self.image, "b.jpg"))
        self.assertFalse(np.array_equal(first, second))

    def test_invalid_severity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ImageDegradation("gaussian_blur", severity=0)


if __name__ == "__main__":
    unittest.main()
