# Đồ án giữa kỳ — Phân đoạn u não cổ điển trên MRI bằng Otsu + Hậu xử lý (đánh giá định lượng Dice/IoU)

> **Mục tiêu:** Xây dựng một pipeline **phân đoạn u não** thuần cổ điển (phương pháp chính: **multi-Otsu + hậu xử lý**), đánh giá định lượng bằng Dice/IoU, và **chứng minh giá trị của từng bước** bằng cách so với baseline Otsu thuần (nghiên cứu loại trừ — ablation). Khép lại bằng phân tích giới hạn của classical → động lực cho phương pháp AI ở project cuối kỳ.
>
> **Phương pháp:** Hoàn toàn cổ điển (không AI) — median filter, Otsu / multi-Otsu, morphology (opening/closing + giữ thành phần liên thông lớn nhất).
>
> **Định vị 2 cấu hình (KHÔNG phải 2 đối thủ ngang hàng):**
> - **`otsu_postproc` = phương pháp chính** — pipeline phân đoạn đầy đủ.
> - **`otsu` = baseline / ablation** — bản rút gọn bỏ bước hậu xử lý, dùng để đo *từng thành phần đóng góp bao nhiêu* vào Dice (morphology + keep-largest-component: 0.36→0.42), chứ không nhằm "thi đấu".
>
> Đây là lập luận thay cho khung "so sánh nhiều phương pháp": câu hỏi không còn là *"sao chỉ có 2?"* mà là *"có / không một thành phần thì kết quả đổi thế nào?"* — đúng bản chất ablation.
>
> **⚠️ Rút gọn phạm vi:** đồ án ban đầu so sánh 4 phương pháp; nay tập trung vào **1 pipeline chính (`otsu_postproc`) + 1 baseline (`otsu`)** để giảm khối lượng và đóng khung chặt hơn. `region_growing` và `marker-based watershed` (cùng `src/watershed_seg.py`, Sobel, distance transform) **đã bỏ**. Các mục G1/G3 liên quan tới chúng dưới đây giữ lại để ghi lịch sử, đánh dấu ~~gạch~~ là đã loại.
>
> **Deliverable:** Jupyter notebook (theory + code interleaved + output nhúng) + report.
>
> **Nguyên tắc:** Ưu tiên **sản phẩm cuối chạy được, giải thích & phân tích được một cách logic**. Mọi lựa chọn thiết kế (threshold, chọn kênh, tham số morphology) phải có cơ sở và đo được, ghi lại để biện luận trong report.

---

## Quy ước phối hợp Người ↔ Claude Code

- `[ ]` chưa làm · `[x]` xong · `[~]` đang làm
- **(Người)** = việc cần bạn quyết định / chuẩn bị / viết nội dung học thuật
- **(CC)** = việc giao cho Claude Code (viết code, refactor, chạy thử)
- **(Cả hai)** = review chung, quyết định dựa trên kết quả

---

## Giai đoạn 0 — Thiết lập môi trường & dữ liệu

- [x] **(Người)** Quyết định dataset cuối cùng — đề xuất: **LGG Segmentation Dataset** (Buda et al., Kaggle) vì ảnh 2D `.tif` có sẵn mask, nhẹ, hợp project classical
- [x] **(Người)** Tải dataset về, kiểm tra cấu trúc thư mục (ảnh + mask tương ứng)
- [x] **(CC)** Tạo cấu trúc thư mục dự án: `data/`, `notebooks/`, `src/`, `results/`, `figures/`
- [x] **(CC)** Viết `requirements.txt`: numpy, scipy, scikit-image, opencv-python, matplotlib, pandas, nbformat
- [x] **(CC)** Viết hàm `load_dataset()` đọc cặp (ảnh, mask), trả về list/generator; xử lý mask nhị phân hóa (>0 → 1) → `src/dataset.py`
- [x] **(Cả hai)** Chạy thử load 5 ảnh + mask, hiển thị overlay để xác nhận dữ liệu đúng → `figures/phase0_overlay_check.png` (chờ bạn xem & xác nhận)

## Giai đoạn 1 — Dựng pipeline cổ điển nền tảng bằng Python 3

> Dựng pipeline cổ điển (preprocess → edge → threshold → watershed) sạch bằng Python 3 + OpenCV 4 / scikit-image. Mọi chỗ ad-hoc thay bằng lựa chọn có cơ sở để G3–G5 phân tích được.

- [x] **(CC)** Viết `src/preprocessing.py`: grayscale, median filter (kernel 5×5)
- [x] ~~**(CC)** Viết hàm Sobel gradient → `sobel_gradient()`~~ **(đã bỏ cùng watershed)**
- [x] ~~**(CC)** Viết `src/watershed_seg.py`: marker-based watershed~~ **(đã bỏ — file đã xóa)**
- [x] ~~**(Cả hai)** Chạy pipeline watershed trên 1 ảnh, 12 panel → `figures/phase1_watershed_pipeline.png`~~ **(đã bỏ — figure đã xóa)**
- [x] **(CC)** Thêm `to_intensity(image, channel="flair")` vào `preprocessing.py`: chọn **kênh FLAIR** làm input segment (u nổi nhất, đo được +47 tương phản so với −9/−2 của pre/post-contrast); benchmark G3 cố định FLAIR cho mọi phương pháp để công bằng
- [ ] **(Người)** Ghi lại các **quyết định thiết kế** để giải thích trong report (đều có số liệu trong code/commit): (a) dùng FLAIR thay vì gộp 3 kênh (+47 vs −9/−2); (b) multi-Otsu 3 lớp thay Otsu 2 lớp/threshold cứng 50 (0.33 vs 0.17); (c) hạt giống region growing = đỉnh FLAIR sau làm mờ Gauss; (d) keep-largest-component trong otsu_postproc (0.42 vs 0.36); (e) bước ad-hoc `sobelx+sobely+gray` của bản gốc **đã bỏ** sau thực nghiệm (cộng cạnh Sobel chỉ làm sáng viền sọ/não & khoét rỗng ruột vùng đặc → thừa với marker-based watershed; Dice 0.444→0.459, phương sai giảm), nay ngưỡng thẳng trên ảnh đã median; các chỗ ad-hoc *còn lại* (`0.7*dist.max`, số vòng morphology) và vì sao không tổng quát

## Giai đoạn 2 — Khung đánh giá định lượng (lõi học thuật)

> Đây là thứ biến "demo" thành "nghiên cứu". Tái sử dụng framework IoU từ notebook edge detection trước đây.

- [x] **(CC)** Viết `src/metrics.py`: hàm `dice(pred, gt)` và `iou(pred, gt)` cho mask nhị phân, có xử lý edge case (mẫu số = 0 → cả hai rỗng trả 1.0, một rỗng trả 0.0)
- [x] **(CC)** Viết unit test nhỏ cho metrics → `tests/test_metrics.py` (9 test, chạy `venv/bin/python -m tests.test_metrics`, **9/9 pass**)
- [x] **(CC)** Viết hàm `evaluate(pred_masks, gt_masks)` trả về Dice/IoU trung bình ± std trên cả tập (kèm vector điểm từng ảnh để vẽ box plot ở G4)
- [x] **(Cả hai)** Xác nhận hướng của pred (vùng u = foreground) khớp với mask ground truth — quy ước **foreground = 1** nhất quán: `dataset.py` nhị phân hóa mask, `segment()` trả u=1, `metrics` nhị phân hóa `>0`

## Giai đoạn 3 — Cài đặt pipeline chính + baseline ablation

> Cài đặt phương pháp chính `otsu_postproc` và baseline `otsu` (bản bỏ hậu xử lý) để đo đóng góp của từng bước (rút gọn từ 4 phương pháp ban đầu).

- [x] **(CC)** **Baseline — Otsu thresholding** — dùng **multi-Otsu 3 lớp** (nền/não/u), lấy lớp sáng nhất; Otsu 2 lớp toàn cục chỉ tách não/nền nên bỏ (đo Dice 0.17 vs 0.33). Đây là **baseline ablation** (pipeline chính bỏ bước hậu xử lý) → `segment_otsu`
- [x] **(CC)** **Phương pháp chính — Thresholding + hậu xử lý** — multi-Otsu + opening/closing + **giữ thành phần liên thông lớn nhất** (loại mô sáng lạc, Dice 0.36→0.42) → `segment_otsu_postproc`
- [x] ~~**(CC)** **Region growing** → `segment_region_growing`~~ **(đã bỏ)**
- [x] ~~**(CC)** **Marker-based watershed** → `segment_watershed`~~ **(đã bỏ)**
- [x] **(Người)** **Thay threshold cố định 50 bằng Otsu** — đã chốt làm nguyên tắc thiết kế ở G1 (tránh tham số tùy tiện); G3 chỉ áp dụng nhất quán cho cả hai phương pháp
- [x] **(CC)** Viết interface thống nhất: mỗi phương pháp là một hàm `segment(image) -> binary_mask`, **đều nhận input FLAIR + median 5×5** để so sánh công bằng → registry `METHODS` trong `src/methods.py`

## Giai đoạn 4 — Thực nghiệm & bảng kết quả

- [x] **(CC)** Viết vòng lặp benchmark: chạy cả 2 phương pháp trên 40 ảnh (mỗi bệnh nhân 1 lát có u), thu Dice/IoU → `src/benchmark.py`
- [x] **(CC)** Xuất bảng kết quả (pandas DataFrame → markdown table) → `results/phase4_summary.md` + `phase4_summary.csv` + `phase4_per_image.csv`
- [x] **(CC)** Vẽ box plot Dice theo phương pháp → `figures/phase4_dice_boxplot.png`
- [x] **(CC)** Vẽ hình định tính 4 ca trải theo kích thước u (FLAIR+GT | từng phương pháp) → `figures/phase4_qualitative.png`; thêm ca **Dice cao nhất** mỗi phương pháp → `figures/phase4_best_cases.png` (otsu_postproc tới ~0.9 — giới hạn trên của classical khi u to, sáng rõ)
- [~] **(Cả hai)** Đọc bảng, xác định mức đóng góp của morphology & pattern — **CC quan sát (chờ bạn xác nhận), 40 ca FLAIR:** pipeline chính otsu_postproc đạt Dice **0.424**±0.41 (IoU 0.364), baseline otsu **0.328**±0.27 → **morphology + keep-largest đóng góp ~+0.10 Dice**. Pattern: cả hai phương sai *rất* lớn (không ổn định); keep-largest-component nâng trung bình nhưng cũng tăng phương sai (chọn nhầm khối → Dice 0) — chính là giới hạn dẫn sang G5

## Giai đoạn 5 — Phân tích giới hạn (cầu nối sang AI)

- [x] **(CC)** Tự động chọn 3 ca có Dice thấp nhất cho mỗi phương pháp, lưu hình → `figures/phase5_worst_cases.png` (`plot_worst_cases` trong `src/benchmark.py`)
- [ ] **(Người)** Viết phân tích thất bại: tại sao thất bại (u nhỏ, tương phản thấp, nhiễu Rician, bias field, chọn nhầm thành phần lớn nhất)
- [ ] **(Người)** Viết đoạn kết nối: giới hạn của classical → động lực cho U-Net ở project cuối kỳ (cùng dataset, so sánh trực tiếp)
- [x] **(CC)** (Tùy chọn) Phân tích độ nhạy tham số → `src/sensitivity.py`, `figures/phase5_sensitivity.png`, `results/phase5_sensitivity.csv`. **Kết quả:** quét `morph_iter` của otsu_postproc — Dice *bền* (gần đơn điệu, bão hòa sau vài vòng), mặc định (2 vòng) ở sát tối ưu (không cherry-pick)

## Giai đoạn 6 — Viết report & hoàn thiện notebook

> Cấu trúc report (mỗi phần map sang markdown cell trong notebook):

- [ ] **(Người)** **1. Đặt vấn đề** — ý nghĩa lâm sàng (glioma, lập kế hoạch xạ trị) + tại sao bài toán khó (tương phản thấp, nhiễu, over-segmentation)
- [ ] **(Người)** **2. Cơ sở lý thuyết** — median filter, Sobel, Otsu, distance transform, watershed; trích Gonzalez & Woods chương 10
- [ ] **(Người)** **3. Phương pháp** — mô tả pipeline phân đoạn chính (otsu_postproc) + baseline ablation (otsu) và mục đích của baseline
- [ ] **(Cả hai)** **4. Thực nghiệm & đánh giá** — dataset, metric, bảng + biểu đồ (kết quả từ Giai đoạn 4)
- [ ] **(Người)** **5. Phân tích giới hạn** — failure cases, độ nhạy tham số
- [ ] **(Người)** **6. Kết luận & hướng phát triển** — classical → U-Net
- [x] **(CC)** Ghép tất cả vào notebook cuối → `Classical_Segmentation_Comparison_Report.ipynb` (25 cell, 6 phần, code import từ `src/`, output đã nhúng). **Phần lý thuyết/biện luận là bản nháp CC, đánh dấu 🔖 chờ (Người) rà.**
- [~] **(Cả hai)** Restart & Run All — CC đã execute bằng `jupyter nbconvert --execute` (**0 lỗi**, 7 ảnh + 1 bảng nhúng); chờ bạn mở xem & xác nhận nội dung
- [ ] **(Người)** Đọc soát lần cuối, kiểm tra trích dẫn nguồn dataset + Gonzalez & Woods

---

## Định nghĩa "Hoàn thành" (Definition of Done)

- [x] Notebook chạy được từ đầu đến cuối không lỗi (Restart & Run All) — `nbconvert --execute` 0 lỗi
- [x] Có bảng Dice/IoU cho 2 phương pháp trên 40 ảnh
- [x] Có biểu đồ so sánh (box plot) + hình định tính overlay (+ best/worst cases)
- [x] Có phân tích giới hạn với failure cases cụ thể (`phase5_worst_cases.png` + độ nhạy tham số) — *văn phân tích chờ (Người)*
- [~] Report đủ 6 phần, có trích dẫn nguồn — khung 6 phần + tài liệu tham khảo đã có; *prose lý thuyết/biện luận là nháp CC, chờ (Người) rà*
- [x] Không dùng bất kỳ thành phần AI/deep learning nào (giữ thuần classical)

---

## Ghi chú gợi ý cho Claude Code

- Dùng `scikit-image` và `OpenCV`, ưu tiên `skimage` cho tính nhất quán với các notebook trước
- Mask ground truth cần nhị phân hóa nhất quán (foreground = u = 1)
- Tách logic ra `src/*.py` để notebook gọn, dễ test; notebook chỉ import + trình bày
- Mọi metric tính trên mask nhị phân, **không** dùng PSNR/SSIM (đó là cho restoration, không phải segmentation)
- Khi benchmark, cố định mọi tham số có thể để so sánh công bằng giữa các phương pháp