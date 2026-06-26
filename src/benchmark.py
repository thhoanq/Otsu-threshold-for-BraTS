"""Vòng lặp thực nghiệm: chấm Dice/IoU cho các phương pháp trên nhiều ảnh.

Cách chạy:

    venv/bin/python -m src.benchmark            # mặc định 40 ca, mỗi bệnh nhân 1 slice
    venv/bin/python -m src.benchmark 20         # giới hạn 20 ca

Sinh ra:
    results/phase4_per_image.csv    — Dice/IoU từng ảnh × từng phương pháp (long form)
    results/phase4_summary.csv      — bảng tổng hợp mean±std
    results/phase4_summary.md       — bảng markdown để dán vào report/notebook
    figures/phase4_dice_boxplot.png — phân phối Dice theo phương pháp
    figures/phase4_qualitative.png  — ảnh gốc | GT | kết quả từng phương pháp
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataset import Sample, _binarize_mask, find_pairs
from src.methods import METHODS
from src.metrics import dice, iou
from src.preprocessing import to_intensity
from skimage.io import imread

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def select_samples(n: int = 40) -> list[tuple[Path, Path]]:
    """Chọn ``n`` ca, **mỗi bệnh nhân một slice** (lát giữa có u) để trải đa dạng.

    Lấy 1 lát/bệnh nhân tránh việc tập test bị thống trị bởi vài bệnh nhân (các
    slice liền kề của cùng một ca rất giống nhau). Lát giữa thường có u to nhất.
    """
    selected: list[tuple[Path, Path]] = []
    for _patient, grp in groupby(find_pairs(), key=lambda p: p[0].parent.name):
        tum = [(i, m) for i, m in grp if _binarize_mask(imread(m)).any()]
        if not tum:
            continue
        selected.append(tum[len(tum) // 2])
        if len(selected) >= n:
            break
    return selected


def run_benchmark(n: int = 40) -> pd.DataFrame:
    """Chạy mọi phương pháp trên ``n`` ca, trả về DataFrame long-form.

    Cột: ``image``, ``patient``, ``gt_px``, ``method``, ``dice``, ``iou``.
    """
    records = []
    pairs = select_samples(n)
    for image_path, mask_path in pairs:
        image = imread(image_path)
        gt = _binarize_mask(imread(mask_path))
        for name, fn in METHODS.items():
            pred = fn(image)
            records.append({
                "image": image_path.name,
                "patient": image_path.parent.name,
                "gt_px": int(gt.sum()),
                "method": name,
                "dice": dice(pred, gt),
                "iou": iou(pred, gt),
            })
    return pd.DataFrame.from_records(records)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp mean±std Dice/IoU theo phương pháp, sắp theo Dice giảm dần."""
    g = df.groupby("method").agg(
        dice_mean=("dice", "mean"), dice_std=("dice", "std"),
        iou_mean=("iou", "mean"), iou_std=("iou", "std"),
    )
    # Giữ thứ tự phương pháp theo registry rồi sắp lại theo Dice để đọc bảng dễ.
    return g.reindex(METHODS.keys()).sort_values("dice_mean", ascending=False)


def _summary_markdown(summary: pd.DataFrame, n: int) -> str:
    lines = [
        f"Kết quả benchmark trên {n} ảnh (mỗi bệnh nhân 1 lát có u), input FLAIR.",
        "",
        "| Phương pháp | Dice (mean ± std) | IoU (mean ± std) |",
        "|---|---|---|",
    ]
    for method, r in summary.iterrows():
        lines.append(
            f"| {method} | {r.dice_mean:.3f} ± {r.dice_std:.3f} "
            f"| {r.iou_mean:.3f} ± {r.iou_std:.3f} |"
        )
    return "\n".join(lines) + "\n"


def plot_boxplot(df: pd.DataFrame, out: Path) -> None:
    """Box plot phân phối Dice theo phương pháp (thứ tự = Dice trung bình giảm dần)."""
    order = df.groupby("method")["dice"].mean().sort_values(ascending=False).index
    data = [df.loc[df.method == m, "dice"].values for m in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=list(order), showmeans=True)
    ax.set_ylabel("Dice")
    ax.set_title("Phân phối Dice theo phương pháp")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _overlay(flair: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Nền FLAIR (xám→RGB), viền GT xanh lá, viền pred đỏ."""
    base = cv2.cvtColor(flair, cv2.COLOR_GRAY2RGB)
    for mask, colour in ((gt, (0, 255, 0)), (pred, (255, 0, 0))):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, contours, -1, colour, 1)
    return base


def plot_qualitative(pairs: list[tuple[Path, Path]], out: Path, k: int = 4) -> None:
    """Hình định tính: chọn ``k`` ca trải theo kích thước u; mỗi hàng 1 ca.

    Cột: FLAIR + GT (xanh) | từng phương pháp (viền pred đỏ vs GT xanh).
    """
    # Đọc gt_px để chọn k ca trải đều theo kích thước u (nhỏ → lớn).
    info = [(ip, mp, int(_binarize_mask(imread(mp)).sum())) for ip, mp in pairs]
    info.sort(key=lambda x: x[2])
    idx = np.linspace(0, len(info) - 1, k).round().astype(int)
    chosen = [info[i] for i in idx]

    methods = list(METHODS.items())
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(k, ncol, figsize=(3.2 * ncol, 3.2 * k))
    for row, (ip, mp, gt_px) in enumerate(chosen):
        image = imread(ip)
        flair = to_intensity(image, "flair")
        gt = _binarize_mask(imread(mp))
        axes[row, 0].imshow(_overlay(flair, np.zeros_like(gt), gt))
        axes[row, 0].set_ylabel(f"u={gt_px}px", fontsize=9)
        if row == 0:
            axes[row, 0].set_title("FLAIR + GT", fontsize=10)
        for col, (name, fn) in enumerate(methods, start=1):
            pred = fn(image)
            axes[row, col].imshow(_overlay(flair, pred, gt))
            label = f"{name}\n" if row == 0 else ""
            axes[row, col].set_title(f"{label}Dice={dice(pred, gt):.2f}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Kết quả định tính (viền GT xanh, viền dự đoán đỏ)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_best_cases(
    df: pd.DataFrame, pairs: list[tuple[Path, Path]], out: Path
) -> None:
    """Mỗi phương pháp: hiển thị ca đạt Dice cao nhất (xem classical tốt nhất tới đâu)."""
    lookup = {ip.name: (ip, mp) for ip, mp in pairs}
    methods = list(METHODS.items())
    fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.8))
    for col, (name, fn) in enumerate(methods):
        sub = df[df.method == name]
        best = sub.loc[sub["dice"].idxmax()]
        ip, mp = lookup[best.image]
        image = imread(ip)
        flair = to_intensity(image, "flair")
        gt = _binarize_mask(imread(mp))
        pred = fn(image)
        axes[col].imshow(_overlay(flair, pred, gt))
        axes[col].set_title(
            f"{name}\n{best.image}\nDice={best.dice:.2f}", fontsize=9)
        axes[col].set_xticks([]); axes[col].set_yticks([])
    fig.suptitle("Ca cao nhất mỗi phương pháp (viền GT xanh, dự đoán đỏ)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_worst_cases(
    df: pd.DataFrame, pairs: list[tuple[Path, Path]], out: Path, k: int = 3
) -> None:
    """Mỗi phương pháp: ``k`` ca Dice thấp nhất (tang chứng cho phân tích thất bại G5).

    Mỗi hàng = một phương pháp; ``k`` cột = ``k`` ca tệ nhất của nó (viền GT xanh,
    dự đoán đỏ, kèm Dice). Bỏ ca slice không có u (gt_px=0 không nằm trong tập này).
    """
    lookup = {ip.name: (ip, mp) for ip, mp in pairs}
    methods = list(METHODS.items())
    fig, axes = plt.subplots(len(methods), k, figsize=(3.2 * k, 3.4 * len(methods)))
    for row, (name, fn) in enumerate(methods):
        worst = df[df.method == name].nsmallest(k, "dice")
        for col, (_, rec) in enumerate(worst.iterrows()):
            ip, mp = lookup[rec.image]
            image = imread(ip)
            flair = to_intensity(image, "flair")
            gt = _binarize_mask(imread(mp))
            axes[row, col].imshow(_overlay(flair, fn(image), gt))
            axes[row, col].set_title(
                f"{name} | Dice={rec.dice:.2f}\n{rec.image[:24]} (u={rec.gt_px}px)",
                fontsize=8)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.suptitle("Ca thấp nhất mỗi phương pháp — viền GT xanh, dự đoán đỏ",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(n: int = 40) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    pairs = select_samples(n)
    n = len(pairs)
    print(f"Chạy benchmark trên {n} ca, {len(METHODS)} phương pháp...")

    df = run_benchmark(n)
    df.to_csv(RESULTS / "phase4_per_image.csv", index=False)

    summary = summarise(df)
    summary.to_csv(RESULTS / "phase4_summary.csv")
    md = _summary_markdown(summary, n)
    (RESULTS / "phase4_summary.md").write_text(md, encoding="utf-8")

    plot_boxplot(df, FIGURES / "phase4_dice_boxplot.png")
    plot_qualitative(pairs, FIGURES / "phase4_qualitative.png")
    plot_best_cases(df, pairs, FIGURES / "phase4_best_cases.png")
    plot_worst_cases(df, pairs, FIGURES / "phase5_worst_cases.png")

    print("\n" + md)
    print("Đã lưu: results/phase4_*.{csv,md}, figures/phase4_*.png")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
