from __future__ import annotations

import cv2
import numpy as np


# Chỉ số kênh trong ảnh LGG 3 kênh (theo README dataset): pre / FLAIR / post-contrast.
_CHANNEL_INDEX = {"pre": 0, "flair": 1, "post": 2}


def to_intensity(image: np.ndarray, channel: str | int = "flair") -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    idx = _CHANNEL_INDEX.get(channel, channel)
    if not isinstance(idx, int) or not 0 <= idx < image.shape[2]:
        raise ValueError(f"channel không hợp lệ: {channel!r}")
    return image[..., idx].astype(np.uint8, copy=False)


def median_filter(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size của median filter phải là số lẻ")
    return cv2.medianBlur(gray, kernel_size)


def preprocess(image: np.ndarray, channel: str | int = "flair", kernel_size: int = 5) -> np.ndarray:
    return median_filter(to_intensity(image, channel), kernel_size)


if __name__ == "__main__":
    from src.dataset import load_dataset

    sample = next(load_dataset(limit=1, tumour_only=True))
    flair = to_intensity(sample.image, "flair")
    median = median_filter(flair, 5)
    print(f"{sample.image_path.name}")
    print(f"  flair   {flair.shape} {flair.dtype} max={flair.max()}")
    print(f"  median  {median.shape} {median.dtype} max={median.max()}")
