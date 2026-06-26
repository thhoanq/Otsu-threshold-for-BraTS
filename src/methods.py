from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label

from src.preprocessing import preprocess


MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MIN_FOREGROUND = 10


def segment_otsu(image: np.ndarray) -> np.ndarray:
    f = preprocess(image, "flair")
    if f.max() < MIN_FOREGROUND:
        return np.zeros(f.shape, dtype=np.uint8)
    try:
        thresholds = threshold_multiotsu(f, classes=3)
    except ValueError:
        return np.zeros(f.shape, dtype=np.uint8)  # không tách được 3 lớp
    return (f > thresholds[-1]).astype(np.uint8)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Giữ thành phần liên thông lớn nhất (u thường là một khối liền)."""
    lab = label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # bỏ nền
    return (lab == sizes.argmax()).astype(np.uint8)


def segment_otsu_postproc(image: np.ndarray, morph_iter: int = 2) -> np.ndarray:
    """Otsu + hậu xử lý: opening (khử đốm) + closing (lấp lỗ) + giữ thành phần lớn nhất.

    Multi-Otsu nhặt cả mô sáng lạc (chất xám, mắt/xoang ở lát nền sọ) ngoài u.
    Bước hậu xử lý: opening/closing làm mịn, rồi **giữ thành phần liên thông lớn
    nhất** loại các đốm sáng rời — đo được Dice tăng ~0.46→0.55 so với chỉ
    opening/closing.

    ``morph_iter`` = số vòng lặp opening/closing (mặc định 2); ``0`` = chỉ
    multi-Otsu + giữ thành phần lớn nhất, không morphology (dùng cho khảo sát G5).
    """
    mask = segment_otsu(image)
    if mask.max() == 0:
        return mask
    if morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=morph_iter)
    return _keep_largest_component(mask)


# Registry: tên phương pháp → hàm segment(image) -> binary_mask
METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "otsu": segment_otsu,
    "otsu_postproc": segment_otsu_postproc,
}


if __name__ == "__main__":
    from src.dataset import load_dataset
    from src.metrics import dice, iou

    sample = next(load_dataset(limit=1, tumour_only=True))
    print(f"{sample.image_path.name} (GT u = {int(sample.mask.sum())} px)")
    for name, fn in METHODS.items():
        pred = fn(sample.image)
        print(f"  {name:15s} mask={int(pred.sum()):6d} px  "
              f"Dice={dice(pred, sample.mask):.3f}  IoU={iou(pred, sample.mask):.3f}")
