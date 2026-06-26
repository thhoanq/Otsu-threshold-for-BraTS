# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **mid-term academic project**: a quantitative comparison study of two *classical* segmentation methods (Otsu, and Otsu + post-processing / "hậu xử lý") for brain-tumour detection on MRI, benchmarked with Dice/IoU against ground-truth masks — using **classical image processing** (OpenCV / scikit-image), not machine learning. *(Originally scoped with four methods; region growing and marker-based watershed have been dropped to reduce workload — only Otsu and Otsu+post-processing remain.)* **`plan.md` is the source of truth for the roadmap** and the Human↔Claude Code division of labour (`(Người)` = the user decides/writes; `(CC)` = delegated to Claude Code; `(Cả hai)` = joint review). It is written in Vietnamese.

The work lives in `src/*.py` (see "Project code" below) — Giai đoạn 0–5 of the plan are implemented and run end-to-end. Every design choice (threshold, channel, morphology params) is justified with a measured number and recorded for the report.

Hard constraint from the plan: **this mid-term stays strictly classical — no AI / deep learning.** A U-Net approach on the same dataset is a *separate, later* final project.

**Progress (as of Giai đoạn 5):** dataset loader, preprocessing, metrics+tests, 2 segmentation methods, benchmark + all figures are done. Remaining is mostly `(Người)` report-writing and the Giai đoạn 6 notebook assembly. Always check the checkboxes in `plan.md` for the current state before starting work.

## Running

- A Python 3 virtualenv lives at `venv/` (git-ignored); `requirements.txt` pins it and **now includes** `scikit-image` 0.26, `scipy` 1.17, `pandas` 3.0 alongside numpy 2.4 / `opencv-python` 4.13 / matplotlib 3.11 + the Jupyter stack. Run modules as packages from the repo root (no `src/__init__.py` — implicit namespace package), e.g.:
  - `venv/bin/python -m src.benchmark [n]` — run the whole benchmark (default 40 images), regenerates `results/phase4_*` and `figures/phase4_*`,`phase5_*`.
  - `venv/bin/python -m tests.test_metrics` — run the metric unit tests (standalone, no pytest needed).
  - `venv/bin/python -m src.methods` — the module's `__main__` is a smoke check on one image.
- **Deliverable notebook:** `notebooks/Classical_Segmentation_Report.ipynb` — the Giai đoạn 6 report (6 sections, markdown theory + code importing from `src/`, outputs embedded). Theory/analysis prose is a CC draft flagged with 🔖 for `(Người)` review. The setup cell `chdir`s back to the repo root and adds it to `sys.path`, so the notebook runs from `notebooks/` while `src` imports and data/figure paths (`kaggle_3m/`, `figures/`) still resolve. Reproduce ("Restart & Run All") with: `venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 notebooks/Classical_Segmentation_Report.ipynb`. (`notebooks/main.ipynb` is an earlier working draft with the same `chdir` guard.)

## Pipeline (the "big picture")

Both methods share one short classical flow (in `src/methods.py`), differing only in the post-threshold cleanup:

1. **Preprocess** — FLAIR intensity channel (`to_intensity`), then `median_filter` (5×5) to remove salt-and-pepper noise.
2. **Threshold** — multi-Otsu (3 classes: bg / brain / tumour), take the brightest class. This is `segment_otsu`.
3. **Post-processing / hậu xử lý** (`otsu_postproc` only) — opening (remove speckle) → closing (fill holes), then **keep the largest connected component** to drop stray bright tissue.

Tuning knob that changes results: the morphology iteration count (`morph_iter`, default 2) — swept in `src/sensitivity.py`. All ad-hoc params are gathered at the top of `methods.py`.

## Project code (`src/`)

The actual deliverable is the `src/` package + the report notebook. Modules (each has a `__main__` smoke check):

- `dataset.py` — `load_dataset()` / `find_pairs()` over `kaggle_3m/`; `Sample(image, mask, …)`. Masks are binarised to {0,1} (`_binarize_mask`, `>0`). `tumour_only=True` skips empty-mask slices.
- `preprocessing.py` — `to_grayscale`, `median_filter` (5×5), and **`to_intensity(image, channel="flair")`** — the single-channel input used by both methods.
- `metrics.py` — `dice`, `iou` (binarise `>0`; both-empty → 1.0, one-empty → 0.0), `evaluate()` → `EvalResult(mean/std + per-image vectors)`. Tested in `tests/test_metrics.py` (9 tests).
- `methods.py` — the two comparison methods (`segment_otsu`, `segment_otsu_postproc`) behind a uniform `segment(image) -> binary_mask` in the **`METHODS`** registry. Both share FLAIR + median-5 preprocessing so only the algorithm differs.
- `benchmark.py` — `select_samples` (1 tumour slice per patient for diversity), `run_benchmark` → long-form DataFrame, `summarise`, and the plotting funcs. Writes `results/phase4_*` and `figures/phase4_*` + `figures/phase5_worst_cases.png`.
- `sensitivity.py` — Giai đoạn 5 parameter sweep of `otsu_postproc` morphology iterations (`morph_iter`). Writes `results/phase5_sensitivity.csv` + `figures/phase5_sensitivity.png`. Finding: Dice saturates after a couple of iterations (robust to this param). `segment_otsu_postproc(.., morph_iter)` carries the optional param purely so this sweep can vary it.

### Key design decisions (each backed by a measured number — these are the report's analysis points)

- **Input = FLAIR channel (index 1), not 3-channel greyscale.** On 30 tumour slices the tumour−background contrast is **+47** in FLAIR vs −9 (pre) / −2 (post). Greyscale dilutes this. All methods use `to_intensity(.., "flair")` for a fair comparison.
- **Otsu = multi-Otsu (3 classes: bg / brain / tumour), take the brightest class.** Plain global 2-class Otsu on FLAIR only splits brain-from-background (segments the whole brain, Dice ≈0.17); multi-Otsu actually localises the tumour (Dice ≈0.33).
- **`otsu_postproc` keeps the largest connected component** after opening/closing — removes stray bright tissue (grey matter, eyes/sinuses on skull-base slices). Raises Dice ≈0.36→0.42 but also increases variance (a wrong largest blob → Dice 0).
- Benchmark numbers move slightly between runs/sample-sets because `select_samples` (1 slice/patient) differs from `load_dataset(limit=N)` (first N slices of the first patients) — keep this in mind when comparing quick experiments to the official table.

## Dataset

`kaggle_3m/` (repo root) is the **LGG Segmentation Dataset** (Buda et al., Kaggle / TCIA), 110 patient folders named `TCGA_<institution>_<patient>_<date>`. Each folder holds 3-channel `.tif` MR slices (`..._<slice>.tif`, channels = pre-contrast / FLAIR / post-contrast) paired with **binary FLAIR-abnormality masks** (`..._<slice>_mask.tif`) — 3929 image/mask pairs total. `data.csv` holds genomic-cluster + patient metadata; `README.md` (inside the folder) documents the format. The folder is currently **untracked and not git-ignored** (only `venv/` is) — it's large, so don't `git add` it wholesale without deciding on a data-handling approach.

This is a **segmentation**-labelled dataset (per-pixel masks), which is what enables the Dice/IoU evaluation in `plan.md`. It replaces the older classification-only `brain_tumor_dataset/` (yes/no) referenced in earlier versions, which is no longer in the repo.

## Roadmap

`plan.md` lays out 7 stages (Giai đoạn 0–6) and is the source of truth — check its checkboxes for live status. **Giai đoạn 0–5 are implemented** (data setup; clean Python-3 classical pipeline; Dice/IoU framework + tests; the 2 comparison methods behind `segment(image) -> binary_mask`; benchmark loop + table/box-plot/qualitative/best/worst figures). What remains: a few `(Người)` report sections + `(Cả hai)` confirmations, the optional parameter-sensitivity analysis, and **Giai đoạn 6** — assembling the final notebook (markdown theory cells interleaved with code that imports from `src/`, outputs embedded, Restart & Run All clean).
