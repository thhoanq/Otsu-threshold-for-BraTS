from __future__ import annotations

from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
from skimage.io import imread

# Thư mục dataset mặc định: <gốc repo>/kaggle_3m
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "kaggle_3m"


class Sample(NamedTuple):
    """Một cặp ảnh/mask cùng đường dẫn để truy vết."""
    image: np.ndarray      # ảnh MR, shape (H, W, 3), dtype gốc của file .tif
    mask: np.ndarray       # mask nhị phân {0, 1}, shape (H, W), dtype uint8
    image_path: Path
    mask_path: Path


def find_pairs(root: str | Path = DEFAULT_ROOT) -> list[tuple[Path, Path]]:
    """Quét dataset, trả về danh sách (image_path, mask_path) đã sắp xếp.

    Bỏ qua các file mask khi liệt kê ảnh, và chỉ giữ ảnh có mask tương ứng tồn tại.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {root}")

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(root.rglob("*.tif")):
        if image_path.stem.endswith("_mask"):
            continue
        mask_path = image_path.with_name(f"{image_path.stem}_mask.tif")
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Nhị phân hóa mask về {0, 1} (uint8): mọi pixel > 0 thành 1."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def load_dataset(
    root: str | Path = DEFAULT_ROOT,
    limit: int | None = None,
    tumour_only: bool = False,
) -> Iterator[Sample]:
    """Generator trả về từng :class:`Sample` (ảnh, mask nhị phân, đường dẫn).

    Parameters
    ----------
    root
        Thư mục gốc dataset. Mặc định ``kaggle_3m/`` ở gốc repo.
    limit
        Nếu đặt, chỉ yield tối đa ``limit`` cặp đầu tiên (tiện thử nhanh).
    tumour_only
        Nếu True, bỏ qua các slice có mask rỗng (toàn 0) — chỉ giữ slice có u.
    """
    count = 0
    for image_path, mask_path in find_pairs(root):
        if limit is not None and count >= limit:
            break
        mask = _binarize_mask(imread(mask_path))
        if tumour_only and not mask.any():
            continue
        image = imread(image_path)
        yield Sample(image=image, mask=mask, image_path=image_path, mask_path=mask_path)
        count += 1


if __name__ == "__main__":
    pairs = find_pairs()
    print(f"Tìm thấy {len(pairs)} cặp ảnh/mask trong {DEFAULT_ROOT}")
    for sample in load_dataset(limit=3):
        print(
            f"  {sample.image_path.name}: image {sample.image.shape} {sample.image.dtype}, "
            f"mask {sample.mask.shape} unique={np.unique(sample.mask).tolist()}"
        )
