"""Độ đo đánh giá segmentation nhị phân: Dice và IoU.

Mask nhị phân, **foreground = u = 1**. Mọi đầu vào được
nhị phân hóa về bool (``> 0``) trước khi tính, nên truyền mask {0,1} hay {0,255}
đều cho cùng kết quả.

Edge case mẫu số = 0 (cả pred lẫn gt đều rỗng — tức "không có u" và đoán đúng là
không có u): quy ước trả **1.0** (khớp hoàn hảo). Nếu chỉ một trong hai rỗng →
**0.0**. Đây là quy ước phổ biến cho Dice/IoU trên ảnh y tế.

- Dice (F1) = ``2|A∩B| / (|A| + |B|)``
- IoU (Jaccard) = ``|A∩B| / |A∪B|``
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np


def _to_bool(mask: np.ndarray) -> np.ndarray:
    """Nhị phân hóa về bool, foreground = mọi pixel > 0."""
    return np.asarray(mask) > 0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hệ số Dice giữa mask dự đoán và ground truth. Trả về [0, 1]."""
    p, g = _to_bool(pred), _to_bool(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred và gt khác shape: {p.shape} vs {g.shape}")
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0  # cả hai rỗng → đoán đúng "không có u"
    intersection = np.logical_and(p, g).sum()
    return float(2.0 * intersection / total)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hệ số IoU (Jaccard) giữa mask dự đoán và ground truth. Trả về [0, 1]."""
    p, g = _to_bool(pred), _to_bool(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred và gt khác shape: {p.shape} vs {g.shape}")
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0  # cả hai rỗng
    return float(intersection / union)


class EvalResult(NamedTuple):
    """Thống kê Dice/IoU trên cả tập, kèm vector từng ảnh để vẽ box plot."""

    dice_mean: float
    dice_std: float
    iou_mean: float
    iou_std: float
    dice_scores: np.ndarray   # Dice từng ảnh, shape (N,)
    iou_scores: np.ndarray    # IoU từng ảnh, shape (N,)
    n: int

    def __str__(self) -> str:
        return (
            f"Dice {self.dice_mean:.3f} ± {self.dice_std:.3f} | "
            f"IoU {self.iou_mean:.3f} ± {self.iou_std:.3f} (n={self.n})"
        )


def evaluate(
    pred_masks: Iterable[np.ndarray],
    gt_masks: Iterable[np.ndarray],
) -> EvalResult:
    """Tính Dice/IoU trung bình ± std trên cả tập ảnh.

    ``pred_masks`` và ``gt_masks`` phải cùng độ dài, tương ứng từng cặp. Trả về
    :class:`EvalResult` (mean/std + vector điểm từng ảnh để vẽ phân phối).
    """
    dice_scores: list[float] = []
    iou_scores: list[float] = []
    for pred, gt in zip(pred_masks, gt_masks):
        dice_scores.append(dice(pred, gt))
        iou_scores.append(iou(pred, gt))

    if not dice_scores:
        raise ValueError("Không có cặp mask nào để đánh giá (tập rỗng)")

    d = np.array(dice_scores, dtype=float)
    j = np.array(iou_scores, dtype=float)
    return EvalResult(
        dice_mean=float(d.mean()), dice_std=float(d.std()),
        iou_mean=float(j.mean()), iou_std=float(j.std()),
        dice_scores=d, iou_scores=j, n=len(d),
    )
