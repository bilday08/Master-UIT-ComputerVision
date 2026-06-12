# Kế hoạch thực hiện đồ án 1 tuần — Đề tài 1: Visual Anomaly Detection (VAND / MVTec AD)

> **Môn học:** Thị giác máy tính (Computer Vision) — Cao học UIT
> **Đề tài:** Visual Anomaly Detection — VAND Challenge trên nền tảng **MVTec AD / VisA**
> **Baseline:** Thư viện **Anomalib** (open-edge-platform) + thuật toán **PatchCore**
> **Phần mở rộng (cộng điểm):** Nâng cấp backbone lên **DINOv2** (model `Dinomaly`) và/hoặc **WinCLIP zero-shot**
> **Nhân sự:** 3 thành viên — **Đúng 7 ngày (Ngày 1 → Ngày 7)**
> **Môi trường:** Google Colab (GPU T4 miễn phí) hoặc GPU local ≥ 8GB

---

## 0. Mục lục

1. [Tổng quan & mục tiêu đồ án](#1-tổng-quan--mục-tiêu-đồ-án)
2. [Phân tích sâu đề tài (hiểu sâu — cộng điểm)](#2-phân-tích-sâu-đề-tài-hiểu-sâu--cộng-điểm)
3. [Ánh xạ yêu cầu giảng viên → nhiệm vụ](#3-ánh-xạ-yêu-cầu-giảng-viên--nhiệm-vụ)
4. [Phân công vai trò 3 thành viên](#4-phân-công-vai-trò-3-thành-viên)
5. [Kế hoạch chi tiết 7 ngày](#5-kế-hoạch-chi-tiết-7-ngày)
6. [Cấu trúc slide báo cáo (đúng 4 phần)](#6-cấu-trúc-slide-báo-cáo-đúng-4-phần)
7. [Phụ lục kỹ thuật — lệnh & code đã kiểm chứng](#7-phụ-lục-kỹ-thuật--lệnh--code-đã-kiểm-chứng)
8. [Rủi ro & phương án dự phòng](#8-rủi-ro--phương-án-dự-phòng)
9. [Tiêu chí hoàn thành (Definition of Done)](#9-tiêu-chí-hoàn-thành-definition-of-done)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Tổng quan & mục tiêu đồ án

**Bài toán:** Phát hiện và định vị bất thường (defect) trên ảnh sản phẩm công nghiệp. Mô hình **chỉ được huấn luyện trên ảnh "bình thường" (good)** — không/rất ít mẫu lỗi (chế độ *unsupervised / one-class / cold-start*). Đầu ra gồm:
- **Image-level score:** điểm bất thường mức ảnh (ảnh này có lỗi hay không?).
- **Pixel-level localization:** bản đồ nhiệt (heatmap) khoanh vùng lỗi mức pixel (lỗi nằm ở đâu?).

**Mục tiêu cụ thể sau 1 tuần:**

| # | Mục tiêu | Trạng thái đích |
|---|----------|-----------------|
| M1 | Chạy thành công baseline PatchCore (Anomalib) trên MVTec AD | Có bảng AUROC ≥ 6 category + ảnh heatmap |
| M2 | Thực hiện ≥ 1 phần mở rộng (DINOv2 hoặc WinCLIP) | Có bảng ablation so sánh với baseline |
| M3 | Phân tích chi tiết dự đoán đúng/sai (TP/FP/TN/FN) | Có confusion matrix + example grid + giải thích nguyên nhân |
| M4 | Hoàn thiện slide báo cáo đủ 4 phần + demo | File `.pptx`/`.pdf` + video demo dự phòng |

**Phạm vi làm việc:** Toàn bộ trong thư mục `Master-UIT-ComputerVision/`. Dataset & kết quả chạy lưu trong `datasets/` và `results/` (đặt trong `.gitignore`).

---

## 2. Phân tích sâu đề tài (hiểu sâu — cộng điểm)

> Phần này là "xương sống" để đạt yêu cầu **"phân tích sâu ở mức hiểu sâu"** và phần **giải thích nguyên nhân đúng/sai** mà giảng viên yêu cầu. Cả nhóm cần nắm vững trước khi báo cáo.

### 2.1. Cơ chế PatchCore (vì sao chạy nhanh, vì sao đúng/sai)

PatchCore (Roth et al., **CVPR 2022**) hoạt động qua 4 bước, **không cần lan truyền ngược (no backprop)**:

1. Trích **patch-feature** từ một CNN đóng băng (mặc định **WideResNet-50** = `wide_resnet50_2`, lấy ở `layer2` + `layer3`) trên tập ảnh *good*.
2. **Coreset subsampling** (greedy) nén memory bank còn ~1–10% (mặc định `coreset_sampling_ratio=0.1`).
3. Lúc test: tính **khoảng cách tới láng giềng gần nhất (nearest-neighbour, `num_neighbors=9`)** từ patch-feature ảnh test tới memory bank.
4. Khoảng cách lớn ⇒ **anomaly score** cao ⇒ sinh heatmap mức pixel.

⇒ "Train" chỉ là **1 lượt forward (`max_epochs=1`)** để dựng memory bank → vài phút/category trên GPU T4.

### 2.2. Hai giới hạn cốt lõi sinh ra lỗi (dùng cho phần "tại sao")

| Giới hạn | Hệ quả | Sinh lỗi loại |
|----------|--------|----------------|
| **(i) Memory-bank nearest-neighbour + coreset** | Vân *good* hiếm không có trong coreset → khoảng cách NN lớn; lỗi nhỏ → khoảng cách NN không đủ lớn | FP (vân hiếm), FN (lỗi nhỏ) |
| **(ii) So khớp cục bộ theo patch** | Không mô hình hóa quan hệ toàn cục/ngữ nghĩa | **Mù với logical anomaly** (sai số lượng/vị trí/tổ hợp — MVTec LOCO) |
| **(iii) Domain gap backbone ImageNet** | Feature không tối ưu cho miền công nghiệp; nhạy xoay/ánh sáng | Category transfer kém → động cơ để **đổi sang DINOv2** |

### 2.3. Danh mục kiểu thất bại theo category (khung A–E)

| Mã | Kiểu thất bại | Category điển hình | Nguyên nhân gốc |
|----|---------------|--------------------|------------------|
| **A** | Lỗi nhỏ/độ tương phản thấp → **FN** | `screw`, `grid`, `transistor` | Độ phân giải không gian thấp của patch-feature, lỗi vài pixel bị trung bình hóa |
| **B** | Vân bình thường nhưng hiếm → **FP** | `carpet`, `leather`, `wood`, `tile` | Coreset không phủ hết biến thể vân tự nhiên |
| **C** | Logical anomaly không bắt được → **FN** | (MVTec LOCO) `screw_bag`, `splicing_connectors` | PatchCore chỉ so khớp cục bộ, mù với bố cục toàn cục |
| **D** | Xoay/alignment/ánh sáng → **FP/FN** | `screw`, `metal_nut`, `hazelnut` | PatchCore không bất biến xoay; dịch phân phối feature |
| **E** | Backbone ImageNet transfer kém | `grid`, một số `tile` | Domain gap → động cơ đổi DINOv2 |

**Khung viết "tại sao" cho mỗi ảnh ví dụ:** (1) ghi score & ngưỡng, (2) heatmap có trùng GT mask không, (3) gán vào kiểu A–E, (4) quy về giới hạn (i)/(ii)/(iii).

---

## 3. Ánh xạ yêu cầu giảng viên → nhiệm vụ

| Yêu cầu của giảng viên | Nhiệm vụ trong kế hoạch | Ngày | Phụ trách chính |
|------------------------|--------------------------|------|------------------|
| Chọn 1 benchmark/dataset (MVTec AD/VisA) trong khung 2022–2025 | Mục 5 — Ngày 1 (chốt MVTec AD + đối chiếu VAND CVPR 2023/2024) | 1 | TV1 |
| Chạy thử code baseline (Anomalib/PatchCore, GPU/Colab) | Mục 5 — Ngày 2–3 + [Phụ lục 7.1](#71-baseline-patchcore-trên-mvtec-ad) | 2–3 | TV2 |
| **[Cộng điểm]** Đề tài có tính mở rộng & ứng dụng (DINOv2/WinCLIP/ứng dụng) | Mục 5 — Ngày 3–4 + [Phụ lục 7.2](#72-phần-mở-rộng-dinov2-dinomaly--winclip-zero-shot) | 3–4 | TV1 |
| **[Cộng điểm]** Phân tích sâu ở mức hiểu sâu | Mục 2 + Mục 5 — Ngày 4–5 | 4–5 | Cả nhóm |
| Slide P1: Giới thiệu bài toán + Benchmark/Dataset (quy mô, nhãn, metric) | [Slide 1–3](#6-cấu-trúc-slide-báo-cáo-đúng-4-phần) | 5 | TV3 |
| Slide P2: Baseline + môi trường cài đặt + chạy thử (demo/inference/eval) | [Slide 4–7](#6-cấu-trúc-slide-báo-cáo-đúng-4-phần) | 5–6 | TV3 + TV2 |
| Slide P3: Kết quả + phân tích ví dụ đúng/sai + nguyên nhân | [Slide 8–10](#6-cấu-trúc-slide-báo-cáo-đúng-4-phần) | 5–6 | TV3 |
| Slide P4: Giới hạn benchmark/mô hình + hướng mở rộng/ứng dụng | [Slide 11–13](#6-cấu-trúc-slide-báo-cáo-đúng-4-phần) | 6 | TV1 + TV3 |

---

## 4. Phân công vai trò 3 thành viên

> Điền tên thật vào ô `[Tên]`. Khối lượng được cân đối: **mỗi thành viên đều có nhiệm vụ kỹ thuật + nhiệm vụ báo cáo**, không ai chỉ ngồi không hoặc gánh quá nặng.

| Thành viên | Vai trò chính | Trách nhiệm xuyên suốt |
|------------|---------------|-------------------------|
| **TV1 — [Hạnh] (Trưởng nhóm)** | Điều phối & **Phát triển phần mở rộng** (DINOv2 `Dinomaly` + WinCLIP) | Chốt phạm vi, quản lý tiến độ & Git, nghiên cứu sâu thuật toán, viết phần "hạn chế + mở rộng + ứng dụng" |
| **TV2 — [Dũng]** | **Kỹ sư môi trường & Baseline** | Dựng Colab/GPU, tải dataset, chạy PatchCore trên các category, đảm bảo *reproducibility* (README, requirements), hỗ trợ demo |
| **TV3 — [Kiên]** | **Phân tích kết quả & Soạn slide** | Dựng pipeline trực quan hóa (confusion matrix, ROC, histogram, example grid), phân tích đúng/sai, tổng hợp bảng số, soạn & trau chuốt slide |

**Quy tắc cộng tác:** mỗi ngày 1 stand-up ngắn (15 phút) đầu giờ; commit code/notebook lên nhánh riêng rồi merge vào `main` cuối ngày; mọi con số trên slide phải là **số nhóm tự chạy được** (số từ paper chỉ để đối chiếu, ghi rõ nguồn).

---

## 5. Kế hoạch chi tiết 7 ngày

> Gợi ý lịch: Ngày 1 có thể bắt đầu vào Thứ Hai. Mỗi ngày kết thúc bằng một **🏁 Cột mốc** kiểm tra được. Ước lượng ~4–6 giờ làm việc/người/ngày.

### 🗓️ Ngày 1 — Khởi động, chốt phạm vi & phân tích đề tài

🎯 **Mục tiêu ngày:** Cả nhóm hiểu chung bài toán, chốt dataset/track, dựng khung dự án; môi trường bắt đầu được cài.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV1** | Đọc kỹ `Research Computer Vision.md`; chốt **MVTec AD** (chính) + đối chiếu VAND CVPR 2023/2024 (track zero-shot/few-shot); viết mục "Phân tích sâu" (Mục 2) thành bản nháp; quyết định hướng mở rộng sơ bộ | Bản tóm tắt 1 trang + quyết định track |
| **TV2** | Tạo cấu trúc repo (`notebooks/`, `src/`, `results/`, `.gitignore`); bật Colab GPU T4 (`!nvidia-smi`); cài thử `pip install anomalib`; kiểm tra `torch.cuda.is_available()` | Repo skeleton + Colab notebook chạy được cell cài đặt |
| **TV3** | Lập **bảng dữ kiện dataset & metric** (MVTec AD/VisA/LOCO, AUROC/PRO/F1-max) cho Slide 1–3; thiết kế template slide & quy ước trình bày | Bảng dữ kiện + file slide rỗng có cấu trúc |

🏁 **Cột mốc 1:** Repo dựng xong, Colab thấy GPU, nhóm thống nhất dataset + hướng mở rộng.

---

### 🗓️ Ngày 2 — Cài đặt môi trường & baseline "smoke test"

🎯 **Mục tiêu ngày:** Chạy thành công PatchCore trên **1 category** (`bottle`) đầu-cuối, in ra image/pixel AUROC.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV2** | Tải MVTec AD qua datamodule (auto-download ~5GB); chạy `engine.fit` + `engine.test` PatchCore trên `bottle` (xem [7.1](#71-baseline-patchcore-trên-mvtec-ad)); xác nhận `bottle` image-AUROC ≈ **1.000** / pixel ≈ **0.984** | Log AUROC + checkpoint `bottle` |
| **TV1** | Nghiên cứu sâu 2 hướng mở rộng; **chốt hướng**: ưu tiên `Dinomaly` (DINOv2) làm hướng chính + WinCLIP zero-shot làm hướng phụ; viết spec thí nghiệm ablation | Tài liệu spec ablation |
| **TV3** | Viết script trích `pred_scores / pred_labels / gt_labels`; dựng khung tính **confusion matrix** (sklearn) + histogram score (xem [7.3](#73-phân-tích-kết-quả--tpfptnfn)) | Script phân tích (chạy thử trên kết quả `bottle`) |

🏁 **Cột mốc 2:** Baseline PatchCore chạy được trên `bottle`; script phân tích đọc được output.

---

### 🗓️ Ngày 3 — Baseline đầy đủ + khởi động phần mở rộng

🎯 **Mục tiêu ngày:** Có kết quả baseline trên **≥ 6 category** đại diện; phần mở rộng bắt đầu chạy.

**Bộ category đề xuất (đủ object dễ, object khó-xoay, texture):** `bottle`, `transistor`, `cable`, `screw`, `grid`, `carpet`.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV2** | Chạy PatchCore trên 6 category; bật **visualization** để xuất panel (input + heatmap + GT mask + overlay) vào `results/`; lưu bảng AUROC/pixel-AUROC/PRO | Bảng kết quả baseline + thư mục ảnh `results/` |
| **TV1** | Triển khai `Dinomaly(encoder_name='dinov2reg_vit_base_14')` chạy trên 2–3 category; song song chạy `WinClip()` zero-shot (xem [7.2](#72-phần-mở-rộng-dinov2-dinomaly--winclip-zero-shot)) | Log kết quả DINOv2 + WinCLIP (lần chạy đầu) |
| **TV3** | Chạy pipeline phân tích trên baseline 6 category: confusion matrix, ROC, histogram; bắt đầu phân loại ảnh thành TP/FP/TN/FN | Biểu đồ ROC/histogram + bảng confusion matrix sơ bộ |

🏁 **Cột mốc 3:** Có bảng AUROC baseline 6 category; phần mở rộng cho ra số đầu tiên.

---

### 🗓️ Ngày 4 — Hoàn thiện mở rộng + phân tích đúng/sai chuyên sâu

🎯 **Mục tiêu ngày:** Có **bảng ablation** baseline vs DINOv2 vs WinCLIP; có **example grid TP/FP/FN/TN** kèm giải thích.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV1** | Hoàn tất ablation: DINOv2 (`Dinomaly`) vs WideResNet50 trên cùng category; WinCLIP **zero-shot (k=0)** + **few-shot (k=1,4)**; lập bảng delta image/pixel AUROC + ghi chú VRAM/latency | Bảng ablation hoàn chỉnh |
| **TV2** | (Tùy chọn) chạy thêm VisA hoặc 2–3 category MVTec còn lại để củng cố; rà soát lại số liệu, lưu checkpoint & log có hệ thống | Kết quả bổ sung + log gọn gàng |
| **TV3** | Lập **example grid 4×4** (TP/FP/FN/TN × ảnh gốc/heatmap/GT/overlay); viết phần "tại sao" theo khung A–E → giới hạn (i)/(ii)/(iii); chọn 2–3 ví dụ đắt giá nhất | Example grid + đoạn phân tích nguyên nhân |

🏁 **Cột mốc 4:** Bảng ablation + bộ ví dụ đúng/sai đã có giải thích nguyên nhân.

---

### 🗓️ Ngày 5 — Tổng hợp kết quả & soạn nội dung slide (Phần 1–3)

🎯 **Mục tiêu ngày:** Đóng băng số liệu; dựng nội dung slide cho 3 phần đầu.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV3** | Dựng Slide **Phần 1** (bài toán + dataset + metric) và **Phần 2** (PatchCore + môi trường + demo) và **Phần 3** (kết quả + phân tích đúng/sai) bằng số liệu thật | Slide 1–10 (bản nháp đầy đủ) |
| **TV2** | Cùng TV3 chuẩn bị **Slide 5–7** (lệnh cài đặt, config, screenshot terminal); **quay video demo** `anomalib predict` (dự phòng) | Screenshot + video demo |
| **TV1** | Rà soát tính đúng của mọi con số/khẳng định kỹ thuật trên slide; viết bản nháp **Phần 4** (hạn chế + mở rộng + ứng dụng QC/nông nghiệp/y tế) | Ghi chú review + nháp Phần 4 |

🏁 **Cột mốc 5:** Slide Phần 1–3 hoàn chỉnh nội dung; video demo đã quay.

---

### 🗓️ Ngày 6 — Hoàn thiện slide (Phần 4) + đảm bảo tái lập

🎯 **Mục tiêu ngày:** Bộ slide đủ 4 phần, định dạng đẹp; dự án tái lập được.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV1** | Hoàn thiện **Slide 11–13** (hạn chế benchmark/model; hướng mở rộng: few-shot/logical/foundation model; ứng dụng QC, nông sản, y tế); chuẩn bị 5 câu Q&A dự kiến | Slide Phần 4 + danh sách Q&A |
| **TV2** | Viết `README.md` tái lập (lệnh cài đặt, lệnh chạy, phiên bản Anomalib/timm); chốt `requirements`; dọn `results/` để minh họa | README + dự án chạy lại được |
| **TV3** | Trau chuốt toàn bộ slide (đánh số "Phần 1/2/3/4", thống nhất font/màu, mỗi slide 1 figure "ngôi sao"); ghép slide thành mạch trình bày | Bộ slide hoàn chỉnh `.pptx` |

🏁 **Cột mốc 6:** Slide đủ 4 phần + README tái lập hoàn tất.

---

### 🗓️ Ngày 7 — Tổng duyệt, xuất bản & dự phòng

🎯 **Mục tiêu ngày:** Trình bày trơn tru; nộp bài; có phương án dự phòng demo.

| Thành viên | Nhiệm vụ | Đầu ra |
|------------|----------|--------|
| **TV1** | Điều phối **dry-run** thuyết trình (bấm giờ ~15 phút); phân vai nói; tổng duyệt Q&A | Biên bản dry-run + điều chỉnh |
| **TV2** | Kiểm tra demo live trên máy báo cáo; chuẩn bị **fallback** (video + ảnh output trong slide); export slide ra PDF | Demo sẵn sàng + fallback |
| **TV3** | Sửa slide theo phản hồi dry-run; rà soát chính tả/định dạng; nộp bài (slide + mã nguồn + README) | Bản nộp cuối cùng |

🏁 **Cột mốc 7 (Hoàn thành):** Báo cáo + mã nguồn + demo đã nộp; cả 3 thành viên thuộc phần của mình.

---

### Bảng cột mốc tổng hợp

| Ngày | Cột mốc | Tiêu chí kiểm tra |
|------|---------|-------------------|
| 1 | Khởi động | Repo + Colab GPU + chốt dataset/track |
| 2 | Baseline smoke test | PatchCore chạy `bottle`, AUROC in ra |
| 3 | Baseline đầy đủ | 6 category có kết quả + ảnh heatmap |
| 4 | Mở rộng + phân tích | Bảng ablation + example grid TP/FP/FN/TN |
| 5 | Slide P1–3 | Nội dung 3 phần đầu + video demo |
| 6 | Slide P4 + tái lập | Đủ 4 phần + README chạy lại được |
| 7 | Nộp bài | Dry-run đạt, fallback sẵn sàng, đã nộp |

---

## 6. Cấu trúc slide báo cáo (đúng 4 phần)

**Khuyến nghị 14 slide, ~15 phút trình bày + 3–5 phút Q&A.** Đánh số "Phần 1/2/3/4" ở góc slide để giám khảo thấy đủ cấu trúc.

| Slide | Phần | Nội dung chính | Figure/Table "ngôi sao" |
|-------|------|----------------|--------------------------|
| 0 | — | Tiêu đề, 3 thành viên + MSSV, GVHD | Ảnh teaser: input / ảnh lỗi / heatmap |
| 1 | **P1** | Bài toán VAD là gì & vì sao khó (one-class, image vs pixel level) | Sơ đồ "train good → test phát hiện lỗi" |
| 2 | **P1** | Benchmark/Dataset: MVTec AD (chính) + VisA (phụ) | **Bảng so sánh dataset** (xem dưới) |
| 3 | **P1** | Loại nhãn + metric đánh giá (AUROC, PRO, F1-max) | Đường ROC minh họa |
| 4 | **P2** | Baseline PatchCore — cơ chế memory-bank 4 bước | Pipeline diagram PatchCore |
| 5 | **P2** | Môi trường cài đặt (Anomalib 2.x, PyTorch Lightning) | Code block cài đặt (đã kiểm chứng) |
| 6 | **P2** | Cấu hình & lệnh chạy train/predict | Screenshot terminal "Testing AUROC" |
| 7 | **P2** | Demo run-through (inference + heatmap) | 2–3 cặp input → heatmap → mask |
| 8 | **P3** | Kết quả định lượng (bảng AUROC/pixel/PRO per-category + mean) | **Bảng kết quả** + so sánh PaDiM/FastFlow |
| 9 | **P3** | Ví dụ dự đoán **ĐÚNG** (TP/TN) | Lưới 2×2 input/GT/heatmap |
| 10 | **P3** | Ví dụ dự đoán **SAI** (FP/FN) + giải thích (khung A–E) | 2 FP + 2 FN kèm overlay |
| 11 | **P4** | Hạn chế benchmark & model (bão hòa AUROC, domain gap, logical anomaly) | Bảng "Strength vs Limitation" |
| 12 | **P4** | Mở rộng (DINOv2/WinCLIP/few-shot) + **bảng ablation** | Bảng baseline vs DINOv2 vs WinCLIP |
| 13 | **P4** | Ứng dụng thực tế: QC công nghiệp, nông sản (bệnh trái cây), y tế (tổn thương da) + Kết luận/Q&A | 3 icon ứng dụng |

**Bảng so sánh dataset (cho Slide 2):**

| Thuộc tính | MVTec AD | VisA |
|------------|----------|------|
| Số category | 15 (10 object + 5 texture) | 12 |
| Tổng ảnh | ~5.354 | ~10.821 (9.621 good + 1.200 lỗi) |
| Train | ~3.629 ảnh, **chỉ good** | chủ yếu good |
| Test | ~1.725 (good + lỗi) | ~1.200 (good + lỗi) |
| Nhãn | image label + **pixel mask** | image label + pixel mask |
| Độ phân giải | 700×700 – 1024×1024 | tương tự, nhiều object nhỏ/đa instance |
| Đặc thù | chuẩn de-facto, defect cục bộ | khó hơn (PCB, multi-instance, sai vị trí) |

**Luồng demo an toàn (~2–3 phút):** mở terminal đã cài sẵn → show config → chạy `anomalib predict` trên thư mục ảnh test đã chuẩn bị (**không train live**, chỉ load checkpoint) → mở `results/` so sánh score ảnh good vs ảnh lỗi. **Luôn có video/ảnh dự phòng.**

---

## 7. Phụ lục kỹ thuật — lệnh & code đã kiểm chứng

> ✅ Các lệnh dưới đây nhắm tới **Anomalib 2.5.0** (bản mới nhất trên PyPI, 05/2026) và **timm ≥ 1.0.x**, đã được kiểm chứng đối chiếu mã nguồn/tài liệu chính thức. **Lưu ý quan trọng về phiên bản:** API sau v1.0 dùng `Engine` + `from anomalib.data import MVTecAD` (lớp cũ `MVTec` và file YAML kiểu < v1.0 **không còn dùng được**).

### 7.1. Baseline PatchCore trên MVTec AD

**Cài đặt (1 bước là đủ — `pip install anomalib` đã kéo cả torch + lightning):**

```bash
python --version                  # cần 3.10+ (3.10/3.11/3.12)
pip install anomalib              # đủ cho torch + lightning theo tài liệu hiện hành
# tùy chọn nhóm phụ:  anomalib install --option openvino   (full|core|dev|loggers|notebooks|openvino)
# tùy chọn wheel CUDA: pip install "anomalib[cu126]"  (hoặc [cu118], [cpu])
python -c "import anomalib, torch; print(anomalib.__version__, torch.cuda.is_available())"
```

**Trên Google Colab:** `Runtime > Change runtime type > T4 GPU` → `!nvidia-smi` → `!pip install anomalib` → **restart runtime**. (Đĩa Colab tạm thời — mount Google Drive nếu muốn giữ dataset/`results/` qua các phiên.)

**Train + test PatchCore trên 1 category (Python — Engine API):**

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

datamodule = MVTecAD(
    root="./datasets/MVTecAD",
    category="bottle",
    train_batch_size=8,   # giảm nếu OOM trên GPU 8GB
    eval_batch_size=8,
)
model = Patchcore()        # mặc định: backbone='wide_resnet50_2', layers=('layer2','layer3'),
                           #           coreset_sampling_ratio=0.1, num_neighbors=9
engine = Engine(max_epochs=1)   # PatchCore chỉ cần 1 lượt forward, KHÔNG cần nhiều epoch

engine.fit(datamodule=datamodule, model=model)
print(engine.test(datamodule=datamodule, model=model))   # in image/pixel AUROC
```

**Hoặc dùng CLI (tương đương):**

```bash
anomalib train --model Patchcore --data anomalib.data.MVTecAD \
    --data.category bottle --data.root ./datasets/MVTecAD
# knob bộ nhớ/tốc độ: --model.coreset_sampling_ratio 0.01 --data.train_batch_size 8
```

**Inference / sinh heatmap:**

```bash
anomalib predict --model Patchcore \
    --data_path ./datasets/MVTecAD/bottle/test/broken_large \
    --ckpt_path ./results/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt
# ⚠️ đường dẫn checkpoint có versioning (v0/v1...) — đọc đúng path mà log in ra sau khi train
```

**Số tham chiếu để kiểm tra (PatchCore + WideResNet50):** `bottle` ≈ **1.000** image-AUROC / **0.984** pixel-AUROC; trung bình 15 category ≈ **0.980 / 0.980** (paper: ~99.1% image, lên tới 99.6% ở cấu hình tốt nhất; pixel ~98.1%; PRO ~93.5%). Nếu `bottle` thấp hơn nhiều ⇒ sai cấu trúc thư mục hoặc coreset quá nhỏ.

**Lưu ý lỗi thường gặp:**
- Tutorial cũ dùng `from anomalib.data import MVTec` / file YAML / `get_model()` là **< v1.0**, sẽ lỗi trên 2.x.
- Colab: **restart runtime** sau khi cài để torch/lightning mới nạp đúng.
- OOM trên GPU 8GB: giảm `train_batch_size`/`eval_batch_size` và/hoặc `coreset_sampling_ratio`.
- Pixel-AUROC cần có thư mục `ground_truth/<defect>/*_mask.png` đúng chuẩn MVTec.

### 7.2. Phần mở rộng: DINOv2 (`Dinomaly`) + WinCLIP zero-shot

**Hướng A — DINOv2 (khuyến nghị: dùng model chính thức `Dinomaly`).**

> ⚠️ **Lưu ý kỹ thuật quan trọng (đã kiểm chứng):** Trên **timm ≥ 1.0.x**, đường `features_only=True` *đã hỗ trợ* ViT/DINOv2 và trả về map `(B, C, H, W)`. Tuy nhiên nếu tự truyền chuỗi backbone DINOv2 vào PatchCore thì **không được tái dùng `layers=('layer2','layer3')`** (tên này không tồn tại trên ViT — phải dùng chỉ số `out_indices` lấy từ `model.feature_info`), và **input phải chia hết cho 14 (lý tưởng 518×518)**. Vì các bẫy này, **cách sạch nhất** để có ablation "DINOv2 vs WideResNet" là dùng model `Dinomaly` có sẵn (xử lý token→grid hộ):

```python
from anomalib.data import MVTecAD
from anomalib.models import Dinomaly      # có từ Anomalib ~v2.2
from anomalib.engine import Engine

dm = MVTecAD(category="bottle")
model = Dinomaly(encoder_name="dinov2reg_vit_base_14")   # cũng có _small_14 / _large_14
eng = Engine()
eng.fit(model=model, datamodule=dm)
print(eng.test(model=model, datamodule=dm))
```

Tên timm của DINOv2 (nếu làm hướng nâng cao tự bọc backbone): `vit_{small,base,large,giant}_patch14_dinov2.lvd142m` và biến thể `..._reg4_...` (input gốc 518×518). `Dinomaly` đạt ~**99.6%** image-AUROC trên MVTec AD (CVPR 2025).

**Hướng B — WinCLIP zero-/few-shot (đã tích hợp sẵn trong Anomalib).**

```python
from anomalib.data import MVTecAD
from anomalib.models import WinClip        # đúng chính tả: WinClip
from anomalib.engine import Engine

dm = MVTecAD(category="transistor")
model = WinClip()                          # zero-shot: k_shot=0, KHÔNG cần fit
eng = Engine()
print(eng.test(model=model, datamodule=dm))   # zero-shot gọi test trực tiếp

# Few-shot (k=1,4): CẦN fit để gom k ảnh good tham chiếu
model_fs = WinClip(k_shot=4, class_name="transistor")
eng.fit(model=model_fs, datamodule=dm)
print(eng.test(model=model_fs, datamodule=dm))
```

WinCLIP dùng backbone CLIP `ViT-B-16-plus-240 / laion400m_e31` (tự tải, cần internet). Số tham chiếu: zero-shot ≈ **91.8%** image / **85.1%** pixel AUROC; WinCLIP+ 1-shot ≈ **93.1% / 95.2%**. **Phân tích thất bại:** WinCLIP tốt với object có lỗi "mô tả được bằng ngôn ngữ", yếu với texture mịn (`carpet/grid/tile`) và lỗi cấu trúc tinh vi; nhạy với `class_name`/prompt.

**Bảng ablation cần lập:** hàng = category; cột = `[WideResNet50 img/pix | DINOv2 img/pix | Δ]` + cột ghi chú latency/VRAM (ViT-L ở 518px rất nặng — bắt đầu với small/base). **Giữ nguyên** category, kích thước ảnh, normalization, cấu hình metric giữa các lần chạy để Δ có nghĩa.

### 7.3. Phân tích kết quả — TP/FP/TN/FN

- **Ngưỡng:** Anomalib dùng `F1AdaptiveThreshold` (quét score, chọn ngưỡng cho **F1 lớn nhất**), có `image_threshold` và `pixel_threshold` riêng; kèm normalization (MinMax/CDF) đưa ngưỡng về ~0.5. Nên báo cáo thêm ngưỡng theo **recall mục tiêu** (QC ưu tiên ít bỏ sót) hoặc theo **percentile của score ảnh good**. Lưu ý phương pháp luận: chọn ngưỡng trên chính tập test rồi báo F1 trên đó là **rò rỉ (optimistic bias)** — nên tách validation hoặc nêu rõ giới hạn.
- **Quy ước:** anomaly = **positive (1)**.

| | Dự đoán Anomaly | Dự đoán Normal |
|---|---|---|
| **Thực tế Anomaly** | **TP** (bắt đúng lỗi) | **FN** (bỏ sót — nguy hiểm nhất trong QC) |
| **Thực tế Normal** | **FP** (báo nhầm hàng tốt) | **TN** (đúng là tốt) |

Chỉ số dẫn xuất: Precision, Recall (=TPR), F1, Specificity, **FNR = FN/(TP+FN)** (tỷ lệ bỏ sót), và FPR cho ROC.

- **Lấy số liệu:** trích `pred_scores / pred_labels / gt_labels` từ output → `sklearn.metrics.confusion_matrix` + `ConfusionMatrixDisplay`; làm **per-category** và **gộp**.
- **Trực quan hóa:** bật visualization callback → Anomalib tự xuất panel (input + heatmap + predicted mask + overlay + GT mask) vào `results/...`. Phân loại ảnh test thành 4 nhóm → ghép **example grid 4 hàng (TP/FP/FN/TN) × các cột**.
- **Biểu đồ cần có:** bảng AUROC/pixel/PRO/F1 per-category + mean; **ROC curve**; **PRO curve**; **score histogram** (good vs anomaly + đường ngưỡng — vùng chồng lấn = nơi sinh FP/FN); example grid; (tùy chọn) t-SNE/UMAP feature.

---

## 8. Rủi ro & phương án dự phòng

| Rủi ro | Khả năng | Phương án dự phòng |
|--------|----------|--------------------|
| Colab hết GPU / ngắt phiên | Cao | Mount Drive lưu dataset+results; chạy ít category hơn; có GPU local dự phòng |
| Tải MVTec AD lỗi giữa chừng (~5GB) | Trung bình | Xóa thư mục `datasets/MVTecAD` rồi tải lại; hoặc tải `tar.xz` thủ công từ mvtec.com |
| Mở rộng DINOv2 vướng API/version | Trung bình | Dùng `Dinomaly` (sạch nhất); nếu < v2.2 → nâng cấp hoặc chuyển hẳn sang **WinCLIP** làm hướng mở rộng |
| OOM trên GPU 8GB | Trung bình | Giảm batch size, coreset ratio, kích thước ảnh; bắt đầu ViT small/base |
| Demo live hỏng khi báo cáo | Cao | **Bắt buộc** quay video demo + chèn ảnh output vào slide từ Ngày 5 |
| Số liệu trên slide không tái lập | Trung bình | Chỉ dùng số nhóm tự chạy; README ghi rõ phiên bản & lệnh; số paper ghi nguồn |
| Hết thời gian phần mở rộng | Trung bình | Ưu tiên hoàn thành M1+M3 trước; mở rộng tối thiểu = WinCLIP zero-shot (không cần train) |

---

## 9. Tiêu chí hoàn thành (Definition of Done)

- [ ] **M1 — Baseline:** PatchCore chạy trên ≥ 6 category MVTec AD, có bảng image/pixel AUROC (+PRO) và ảnh heatmap trong `results/`.
- [ ] **M2 — Mở rộng:** Có ≥ 1 phần mở rộng (DINOv2 `Dinomaly` hoặc WinCLIP) với **bảng ablation** so sánh baseline.
- [ ] **M3 — Phân tích:** Confusion matrix (per-category + gộp), score histogram, ROC/PRO, **example grid TP/FP/TN/FN** kèm giải thích nguyên nhân theo khung A–E → giới hạn (i)/(ii)/(iii).
- [ ] **M4 — Báo cáo:** Slide đủ **4 phần** (Giới thiệu+Dataset / Baseline+Môi trường+Demo / Kết quả+Phân tích đúng-sai / Hạn chế+Mở rộng-Ứng dụng) + video demo dự phòng + README tái lập.
- [ ] Mỗi thành viên thuộc và trình bày trơn tru phần của mình (dry-run đạt ~15 phút).
- [ ] Mã nguồn + slide + README đã nộp đúng hạn.

---

## 10. Tài liệu tham khảo

- **MVTec AD** — Bergmann et al., CVPR 2019: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- **VisA** — Zou et al., ECCV 2022 (SPot-the-Difference): <https://github.com/amazon-science/spot-diff> · <https://arxiv.org/abs/2207.14315>
- **MVTec LOCO AD** — Bergmann et al., IJCV 2022: <https://www.mvtec.com/company/research/datasets/mvtec-loco>
- **PatchCore** — Roth et al., CVPR 2022: <https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf>
- **WinCLIP** — Jeong et al., CVPR 2023: <https://arxiv.org/abs/2303.14814>
- **Anomalib** — open-edge-platform: <https://github.com/open-edge-platform/anomalib> · Docs: <https://anomalib.readthedocs.io>
- **VAND Challenge** — CVPR 2023: <https://sites.google.com/view/vand-cvpr23/challenge> · CVPR 2024 (VAND 2.0): <https://sites.google.com/view/vand-2-0-cvpr-2024/challenge>
- **APRIL-GAN** (zero-shot winner VAND 2023): <https://arxiv.org/abs/2305.17382>

> 📌 **Ghi chú trung thực:** một số con số metric/split của VAND 2.0 (CVPR 2024) chưa xác minh đầy đủ từ nguồn gốc — cần kiểm tra lại trên trang workshop chính thức trước khi đưa vào báo cáo. Mọi con số AUROC dùng để báo cáo phải là **số nhóm tự chạy được**; số từ paper chỉ để đối chiếu.

---

*Kế hoạch này được xây dựng dựa trên `Research Computer Vision.md` và đối chiếu/kiểm chứng với tài liệu chính thức của Anomalib, MVTec, PatchCore, WinCLIP và VAND Challenge.*
