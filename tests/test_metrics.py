"""Unit test cho src/metrics.py (Dice / IoU).

Chạy standalone (không cần pytest)::

    venv/bin/python -m tests.test_metrics

Hoặc bằng pytest nếu đã cài: ``pytest tests/``.
"""

from __future__ import annotations

import numpy as np

from src.metrics import dice, evaluate, iou


def test_identical_masks_dice_iou_one():
    """Hai mask trùng hoàn toàn → Dice = IoU = 1."""
    m = np.zeros((8, 8), np.uint8)
    m[2:5, 2:5] = 1
    assert dice(m, m) == 1.0
    assert iou(m, m) == 1.0


def test_disjoint_masks_dice_iou_zero():
    """Hai mask rời nhau (không giao) → Dice = IoU = 0."""
    a = np.zeros((8, 8), np.uint8)
    b = np.zeros((8, 8), np.uint8)
    a[0:3, 0:3] = 1
    b[5:8, 5:8] = 1
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0


def test_both_empty_returns_one():
    """Cả hai mask rỗng (đoán đúng 'không có u') → Dice = IoU = 1."""
    z = np.zeros((8, 8), np.uint8)
    assert dice(z, z) == 1.0
    assert iou(z, z) == 1.0


def test_one_empty_returns_zero():
    """Chỉ một mask rỗng → Dice = IoU = 0."""
    z = np.zeros((8, 8), np.uint8)
    nz = np.zeros((8, 8), np.uint8)
    nz[1:4, 1:4] = 1
    assert dice(z, nz) == 0.0
    assert iou(z, nz) == 0.0
    assert dice(nz, z) == 0.0
    assert iou(nz, z) == 0.0


def test_half_overlap_known_values():
    """Giao một nửa: hai ô 4-pixel chồng nhau 2 pixel → Dice=0.5, IoU=1/3."""
    a = np.zeros((4, 4), np.uint8)
    b = np.zeros((4, 4), np.uint8)
    a[0:2, 0:2] = 1          # 4 pixel
    b[0:2, 1:3] = 1          # 4 pixel, giao 2 pixel (cột 1)
    # Dice = 2*2 / (4+4) = 0.5 ; IoU = 2 / 6 = 1/3
    assert abs(dice(a, b) - 0.5) < 1e-9
    assert abs(iou(a, b) - 1.0 / 3.0) < 1e-9


def test_binarization_invariant_to_scale():
    """Mask {0,255} cho cùng kết quả như {0,1} (nhị phân hóa > 0)."""
    a = np.zeros((4, 4), np.uint8)
    a[0:2, 0:2] = 1
    a255 = a * 255
    assert dice(a, a255) == 1.0
    assert iou(a, a255) == 1.0


def test_shape_mismatch_raises():
    """Khác shape → ValueError."""
    a = np.zeros((4, 4), np.uint8)
    b = np.zeros((4, 5), np.uint8)
    for fn in (dice, iou):
        try:
            fn(a, b)
            raise AssertionError(f"{fn.__name__} đáng lẽ phải raise ValueError")
        except ValueError:
            pass


def test_evaluate_aggregates_correctly():
    """evaluate() tính đúng mean/std và giữ vector điểm từng ảnh."""
    full = np.ones((4, 4), np.uint8)
    empty = np.zeros((4, 4), np.uint8)
    # Cặp 1: trùng hoàn toàn → Dice 1 ; Cặp 2: pred rỗng, gt đầy → Dice 0
    preds = [full, empty]
    gts = [full, full]
    res = evaluate(preds, gts)
    assert res.n == 2
    assert res.dice_mean == 0.5
    assert res.iou_mean == 0.5
    assert list(res.dice_scores) == [1.0, 0.0]


def test_evaluate_empty_raises():
    """Tập rỗng → ValueError."""
    try:
        evaluate([], [])
        raise AssertionError("evaluate([]) đáng lẽ phải raise ValueError")
    except ValueError:
        pass


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
