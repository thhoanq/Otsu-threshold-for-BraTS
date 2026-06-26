"""Phân tích độ nhạy tham số (Giai đoạn 5, tùy chọn).

Quét tham số ``morph_iter`` (số vòng opening/closing) của ``otsu_postproc`` và đo
Dice trung bình thay đổi thế nào — cho thấy phương pháp này bền/mong manh ra sao
theo tham số (điểm phân tích cho report). Cố định mọi thứ khác, chỉ đổi tham số này.

Chạy::

    venv/bin/python -m src.sensitivity [n]      # mặc định 30 ca

Sinh ra:
    results/phase5_sensitivity.csv      — (param, value, dice_mean, dice_std)
    figures/phase5_sensitivity.png      — đường Dice vs tham số (vạch = mặc định)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread

from src.benchmark import select_samples
from src.dataset import _binarize_mask
from src.methods import segment_otsu_postproc
from src.metrics import dice

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# (nhãn tham số, hàm segment(image, value), dải giá trị, giá trị mặc định)
SWEEPS = [
    (
        "otsu + hậu xử lý: số vòng morphology",
        lambda img, v: segment_otsu_postproc(img, morph_iter=v),
        [0, 1, 2, 3, 4, 5],
        2,
    ),
]


def _load_eval_set(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Đọc trước ảnh + mask một lần để quét nhiều giá trị khỏi đọc lại đĩa."""
    data = []
    for image_path, mask_path in select_samples(n):
        data.append((imread(image_path), _binarize_mask(imread(mask_path))))
    return data


def run_sensitivity(n: int = 30) -> pd.DataFrame:
    data = _load_eval_set(n)
    records = []
    for label, fn, values, _default in SWEEPS:
        for v in values:
            scores = [dice(fn(img, v), gt) for img, gt in data]
            records.append({
                "param": label, "value": v,
                "dice_mean": float(np.mean(scores)),
                "dice_std": float(np.std(scores)),
            })
    return pd.DataFrame.from_records(records)


def plot_sensitivity(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(5 * len(SWEEPS), 4.2),
                             squeeze=False)
    for ax, (label, _fn, _values, default) in zip(axes.flat, SWEEPS):
        sub = df[df.param == label]
        ax.plot(sub.value, sub.dice_mean, "o-", color="tab:blue")
        ax.fill_between(sub.value, sub.dice_mean - sub.dice_std,
                        sub.dice_mean + sub.dice_std, alpha=0.15, color="tab:blue")
        ax.axvline(default, ls="--", color="tab:red",
                   label=f"mặc định = {default}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("giá trị tham số")
        ax.set_ylabel("Dice trung bình")
        ax.set_ylim(0, max(0.6, sub.dice_mean.max() + 0.1))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Phân tích độ nhạy tham số (vùng nhạt = ±1 std)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(n: int = 30) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    print(f"Quét độ nhạy tham số trên {n} ca...")
    df = run_sensitivity(n)
    df.to_csv(RESULTS / "phase5_sensitivity.csv", index=False)
    plot_sensitivity(df, FIGURES / "phase5_sensitivity.png")

    for label, _fn, _values, default in SWEEPS:
        sub = df[df.param == label]
        best = sub.loc[sub.dice_mean.idxmax()]
        dflt = sub.loc[sub.value == default].iloc[0]
        print(f"\n{label}")
        print(f"  tốt nhất: value={best.value} → Dice {best.dice_mean:.3f}")
        print(f"  mặc định: value={default} → Dice {dflt.dice_mean:.3f}")
        print(f"  biên độ dao động Dice theo tham số: "
              f"{sub.dice_mean.max() - sub.dice_mean.min():.3f}")
    print("\nĐã lưu results/phase5_sensitivity.csv, figures/phase5_sensitivity.png")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
