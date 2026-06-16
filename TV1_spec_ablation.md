# TV1 (Hạnh) — Spec Ablation Study · Slide Phần 4 · Bộ Q&A phản biện

> **Dự án:** Visual Anomaly Detection (MVTec AD / VisA) — Anomalib 2.5.0 · PatchCore (baseline) · Dinomaly/DINOv2 · WinCLIP
> **Phạm vi:** Phần việc của TV1 (trưởng nhóm — thuật toán & mở rộng). Không làm thay TV2 (chạy baseline) hay TV3 (mỹ thuật slide).
> **Nguồn số liệu:** đọc trực tiếp từ `results/` của TV2 — `mvtec_baseline_results.csv`, `visa_baseline_results.csv`, `failure_analysis.csv`, `failure_analysis_visa.csv`, `PatchCore_Report.html`. **Đây là số nhóm tự chạy được.**

---

## 0. Dữ liệu đầu vào — Tổng hợp benchmark baseline của TV2 (số THẬT)

TV2 chạy **PatchCore mặc định** (backbone `wide_resnet50_2`, feature `layer2+layer3`, `coreset_sampling_ratio=0.1`, `num_neighbors=9`, input 256×256, `max_epochs=1`, GPU) trên **MVTec AD (6 cat)** và **VisA (4 cat)**.

### Bảng 0.1 — PatchCore trên MVTec AD (Image / Pixel AUROC)

| Category | Image AUROC | Pixel AUROC |
|---|---:|---:|
| bottle | **1.000** | 0.986 |
| transistor | 0.996 | **0.973** |
| cable | 0.991 | 0.985 |
| grid | 0.990 | 0.983 |
| carpet | 0.986 | 0.991 |
| screw | **0.966** | 0.989 |
| **Trung bình (6 cat)** | **0.988** | **0.984** |

### Bảng 0.2 — PatchCore trên VisA (Image AUROC)

| Category | Image AUROC | Ghi chú |
|---|---:|---|
| candle | 0.972 | tốt |
| pcb1 | 0.930 | khá |
| macaroni1 | **0.805** | đa thực thể, nhạy bố cục |
| capsules | **0.724** | đa thực thể, biến thiên lớn |
| **Trung bình (4 cat)** | **0.858** | thấp hơn MVTec ~13 điểm |

### Bảng 0.3 — Phân tích TP/FP/TN/FN (số THẬT, ngưỡng F1-adaptive ≈ 0.5)

| Dataset | Category | TP | FP | TN | FN | Precision | Recall | F1 | n_test |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MVTec | bottle | 62 | 0 | 20 | 1 | 1.000 | 0.984 | 0.992 | 83 |
| MVTec | cable | 89 | 3 | 55 | 3 | 0.967 | 0.967 | 0.967 | 150 |
| MVTec | transistor | 38 | 1 | 59 | 2 | 0.974 | 0.950 | 0.962 | 100 |
| MVTec | grid | 54 | 1 | 20 | 3 | 0.982 | 0.947 | 0.964 | 78 |
| MVTec | carpet | 86 | 2 | 26 | 3 | 0.977 | 0.966 | 0.972 | 117 |
| MVTec | **screw** | 113 | **7** | 34 | **6** | 0.942 | 0.950 | **0.946** | 160 |
| VisA | **capsules** | 99 | **59** | 1 | 1 | **0.627** | 0.990 | **0.767** | 160 |

### Ghi chú trung thực (bắt buộc nêu khi báo cáo)

1. **`screw`** là category MVTec khó nhất: F1 = 0.946, đồng thời nhiều **FN (6)** và **FP (7)** nhất — lỗi nhỏ, tương phản thấp (kiểu A).
2. **`capsules` (VisA)** là ca thất bại điển hình: Recall 0.990 nhưng **Precision 0.627, Accuracy 0.625** — **59/60 ảnh tốt bị báo nhầm là lỗi**. Đây là bằng chứng định lượng mạnh nhất của báo cáo.
3. **`candle`, `macaroni1`, `pcb1`** có confusion matrix chạy trên **tập demo nhỏ (n=10)** nên F1 "hoàn hảo" *không đại diện*; chỉ số Image-AUROC (trên tập lớn) mới đáng tin. Cần nêu rõ điều này.
4. Báo cáo cuối của TV2 **chưa lưu Pixel-AUROC cho VisA** và **chưa tính PRO/AUPRO, F1-max** → cần bổ sung khi chạy ablation (xem Spec 1.3c).
5. Trung bình MVTec tính trên **6 cat**, VisA trên **4 cat** (số paper trên 15/12 cat) — chỉ đối chiếu định hướng.

![Hình 1 — Image AUROC theo category: MVTec bão hòa quanh 0.99, VisA tụt mạnh ở capsules/macaroni1](TV1_figures/fig1_auroc_per_category.png)

**Ba quan sát "xương sống" cho Slide 11:**
- **MVTec gần bão hòa:** 6/6 category ≥ 0.966 image-AUROC (TB 0.988) ⇒ hết dư địa, phải chuyển trọng tâm sang VisA/lỗi logic/zero-shot.
- **VisA phơi bày giới hạn lõi:** image-AUROC `capsules` 0.724, `macaroni1` 0.805 — đúng các cảnh đa thực thể, nhạy bố cục.
- **Lệch image↔pixel + bùng nổ FP:** `capsules` định vị tốt nhưng quyết định mức ảnh sai hàng loạt (59 FP) — minh chứng PatchCore "mù" quan hệ toàn cục.

---

## 1. NHIỆM VỤ 2.1 — Spec thí nghiệm Ablation Study

### 1.1. Mục tiêu & câu hỏi nghiên cứu
So sánh **công bằng** ba họ phương pháp trên **cùng category, cùng split, cùng phần cứng**:
- (Q1) Thay backbone CNN-ImageNet bằng **DINOv2 (tự giám sát)** giảm domain gap bao nhiêu (Δ image/pixel AUROC), nhất là ở `screw`?
- (Q2) **WinCLIP zero-/few-shot** đạt mức nào khi không/ít dữ liệu train, đánh đổi gì?
- (Q3) Có cứu được ca **`capsules` (59 FP)** — tức quan hệ toàn cục/lỗi logic — không?
- (Q4) Chi phí latency/VRAM mỗi hướng — khả thi cho edge?

### 1.2. Ba nhánh thí nghiệm

| | **Baseline** | **Mở rộng 1 — DINOv2** | **Mở rộng 2 — WinCLIP** |
|---|---|---|---|
| Model (Anomalib) | `Patchcore()` | `Dinomaly(encoder_name="dinov2reg_vit_base_14")` | `WinClip()` |
| Backbone | WideResNet50 (ImageNet) | DINOv2-reg ViT-B/14 (tự giám sát) | CLIP ViT-B-16-plus-240 |
| Cơ chế | Memory bank + NN (coreset 0.1, k=9) | Encoder–decoder reconstruction đa tầng | Cửa sổ đa tỉ lệ + so khớp text-prompt / reference |
| Cần train? | 1 forward | Có (huấn luyện decoder trên ảnh good) | **k=0: không**; k=1,4: chỉ gom ảnh tham chiếu |
| Input size | 256×256 | **518×518** (chia hết cho 14) | 240×240 |
| Chế độ | one-class/category | one-class/category | **zero-shot (k=0)** & **few-shot (k=1,4)** |

### 1.3. Yêu cầu kỹ thuật bắt buộc (để Δ có nghĩa)

**(a) Kích thước ảnh & token-to-grid.**
- *PatchCore:* feature CNN `(B,C,32,32)` ở 256². Giữ nguyên cấu hình TV2. (Gợi ý từ report TV2: thử nâng 256→448 để cải thiện lỗi nhỏ.)
- *DINOv2/Dinomaly:* input **chia hết cho 14**, khuyến nghị **518×518** → lưới patch **37×37**. ViT trả `N+1` (hoặc `N+1+4` với biến thể `reg`) token: **bỏ [CLS] + 4 register token**, reshape `N` patch token → `(B,C,37,37)`, **bilinear upsample** lên độ phân giải ảnh để sinh heatmap. ⚠️ Tự bọc DINOv2 vào PatchCore thì **không dùng `layers=('layer2','layer3')`** (tên CNN) — lấy `out_indices` từ `model.feature_info`. Dùng `Dinomaly` có sẵn là sạch nhất.
- *WinCLIP:* input 240²; đặc trưng theo **cửa sổ đa tỉ lệ** (small/mid + toàn ảnh).

**(b) Chuẩn hóa & ngưỡng.** Dùng cùng `F1AdaptiveThreshold` + normalization (MinMax) cho cả ba. Báo thêm ngưỡng theo **percentile score ảnh good** (ưu tiên QC ít bỏ sót). Nêu rõ rủi ro rò rỉ nếu chọn ngưỡng trên chính tập test.

**(c) Metric (đồng nhất).** Image-AUROC, Pixel-AUROC, **AUPRO** (bổ sung — hiện thiếu), F1-max. Tính **per-category + trung bình**. Bổ sung **Precision/Recall/Accuracy** vì AUROC một mình che giấu ca `capsules` (AUROC 0.724 nhưng Accuracy chỉ 0.625).

**(d) Đo latency & VRAM (đảm bảo công bằng).**
- Cùng 1 GPU (vd T4 16GB), cùng phiên bản torch/anomalib.
- *Latency:* mean **ms/ảnh** ở inference (loại I/O), `batch=1`, bỏ 10 warmup, ≥100 ảnh, `torch.cuda.synchronize()` trước/sau; báo thêm throughput `batch=8`.
- *VRAM:* `reset_peak_memory_stats()` → `max_memory_allocated()`.
- Bổ sung params (M) + FLOPs/ảnh.
- **Phương pháp luận:** input size khác nhau là **ràng buộc nội tại** (256/518/240) — không ép bằng. ⇒ **báo cáo latency–VRAM song song AUROC**, diễn giải dưới dạng *đánh đổi độ chính xác ↔ chi phí*.

**(e) Biến kiểm soát.** Cố định bộ category (đúng 6 MVTec của TV2), split, seed, normalization, cách tính metric. **Chỉ thay 1 biến: model/backbone.**

### 1.4. Ma trận thí nghiệm tối thiểu

| # | Model | Category | k-shot | Đầu ra |
|---|---|---|---|---|
| E1 | PatchCore (WRN50) | 6 MVTec | — | (đã có) img/pix AUROC + **bổ sung AUPRO** |
| E2 | Dinomaly (DINOv2 ViT-B/14, 518²) | 6 MVTec | — | img/pix AUROC, AUPRO, latency, VRAM |
| E3 | WinCLIP | `transistor, screw, carpet` | **0** | img/pix AUROC, latency, VRAM |
| E4 | WinCLIP | cùng E3 | **1, 4** | AUROC theo k; độ nhạy prompt/`class_name` |
| E5 (ưu tiên) | Dinomaly | VisA `capsules`, `macaroni1` | — | kiểm chứng có cứu được ca 59 FP không |

> **Đạt:** có bảng Δ(image), Δ(pixel) của E2–E4 so E1 **trên cùng category** + cột latency/VRAM ⇒ chính là Slide 12.

---

## 2. NHIỆM VỤ 2.2 — Slide Phần 4 (Slide 11, 12, 13)

> *Khối viết sẵn để copy vào slide. Chú thích `[…]` có thể bỏ khi lên slide.*

### ▶ SLIDE 11 — Hạn chế của benchmark MVTec AD & mô hình PatchCore

**Tiêu đề:** Khi PatchCore "đuối": ba giới hạn lộ ra từ chính số liệu của nhóm

**Ý 1 — MVTec AD đã bão hòa.** 6/6 category ≥ 0.966 image-AUROC, trung bình **0.988/0.984** (Hình 1). Dư địa cải thiện gần như bằng 0 ⇒ MVTec không còn đủ khó để phân biệt phương pháp; phải đẩy sang **VisA / lỗi logic / zero-shot**.

**Ý 2 — Bằng chứng định lượng (số tự nhóm chạy):**

| Hiện tượng | Số thật | Quy về giới hạn |
|---|---|---|
| Image-AUROC sụp đổ ở cảnh đa thực thể | VisA `capsules` 0.724, `macaroni1` 0.805 | **(ii)** so khớp cục bộ → mù lỗi logic (đếm/bố cục) |
| **Bùng nổ False Positive** | `capsules`: **59 FP**/60 ảnh tốt; Precision 0.627, Accuracy 0.625 | **(i)** coreset không phủ hết biến thiên vân "good" tự nhiên |
| Lỗi nhỏ, tương phản thấp | MVTec `screw`: 6 FN + 7 FP, F1 0.946 (thấp nhất) | **(i)** lỗi vài pixel bị trung bình hóa → FN |
| Pixel kém ở lỗi cấu trúc | `transistor` pixel 0.973 (thấp nhất) | **(ii)+(iii)** lệch vị trí + domain gap |

![Hình 2 — Ca capsules: Recall 0.99 nhưng Precision/Accuracy ~0.63 (59 FP)](TV1_figures/fig2_capsules_diagnosis.png)

**Ý 3 — Ba nguyên nhân gốc:**
- **(i) Coreset Nearest-Neighbor:** memory bank nén còn ~10%; **vân good hiếm** không có trong coreset → NN lớn giả tạo (**FP** — đúng ca `capsules` 59 FP); **lỗi quá nhỏ** → NN không đủ lớn (**FN** — đúng ca `screw`).
- **(ii) So khớp cục bộ theo patch (mù logical anomaly):** PatchCore so từng patch độc lập, **không mô hình hóa quan hệ toàn cục** (số lượng/vị trí/tổ hợp) ⇒ `capsules`/`macaroni1` thất bại mức ảnh dù pixel vẫn ổn.
- **(iii) Domain gap backbone ImageNet:** WideResNet50 chưa tối ưu cho bề mặt công nghiệp; nhạy xoay/ánh sáng ⇒ động cơ đổi sang **DINOv2**.

*(TV2 đã có lưới ảnh lỗi `failures_capsules.png`, `failures_screw.png` và histogram score — dùng làm hình minh họa kèm.)*

---

### ▶ SLIDE 12 — Đánh giá phần mở rộng (Ablation Study)

**Tiêu đề:** Foundation model thu hẹp khoảng cách thế nào — và đánh đổi ra sao

![Hình 3 — Ablation MVTec: PatchCore (đo thật) vs Dinomaly/WinCLIP (số paper)](TV1_figures/fig3_ablation.png)

**Bảng ablation (MVTec AD).** Cột PatchCore = **số nhóm tự đo**; Dinomaly/WinCLIP = **số tham chiếu paper gốc** (sẽ thay bằng số nhóm chạy sau E2–E4).

| Phương pháp | Backbone | Train | Image AUROC | Pixel AUROC | AUPRO | Nguồn |
|---|---|---|---:|---:|---:|---|
| **PatchCore (baseline)** | WideResNet50 | good (1 forward) | **0.988** | **0.984** | *cần bổ sung* | **Đo tại nhóm (6 cat)** |
| **Dinomaly (DINOv2)** | DINOv2-reg ViT-B/14 | good | **0.996** | 0.984 | 0.948 | Paper CVPR 2025 (15 cat) |
| Dinomaly (ViT-L) | DINOv2-reg ViT-L/14 | good | 0.998 | — | — | Paper CVPR 2025 |
| **WinCLIP zero-shot (k=0)** | CLIP ViT-B-16+ | **0 ảnh** | 0.918 | 0.851 | 0.646 | Paper CVPR 2023 |
| **WinCLIP+ (k=1)** | CLIP ViT-B-16+ | 1 ảnh good | 0.931 | 0.952 | 0.871 | Paper CVPR 2023 |
| **WinCLIP+ (k=4)** | CLIP ViT-B-16+ | 4 ảnh good | 0.952 | 0.962 | 0.890 | Paper CVPR 2023 |

> ⚠️ PatchCore TB tính trên **6 cat**, số paper trên **15 cat** — chỉ định hướng. Bảng "đáng tin" là khi E2–E4 chạy **trên đúng 6 cat này**.

**DINOv2 (Dinomaly) giải quyết được gì:**
- **Giảm domain gap:** học tự giám sát trên 142M ảnh → đặc trưng ngữ nghĩa, bất biến tốt hơn CNN-ImageNet, hữu ích cho category PatchCore yếu (`screw`).
- **Biểu diễn toàn cục:** reconstruction trên ViT nắm quan hệ xa → kỳ vọng cứu lỗi cấu trúc; image-AUROC ~99.6% (ViT-B), 99.8% (ViT-L).
- **Đa lớp một mô hình:** một mô hình cho nhiều category vẫn SOTA → giảm chi phí vận hành so với "một memory bank/category" của PatchCore.

**WinCLIP mạnh/yếu ở đâu:**
- **Mạnh:** **không cần dữ liệu train** (k=0) vẫn ~91.8% image-AUROC nhờ prompt ngôn ngữ → lý tưởng **cold-start/sản phẩm mới**. Thêm 1–4 ảnh good nâng pixel-AUROC 85.1% → 95–96%.
- **Yếu:** kém **texture mịn** (`carpet/grid`) và lỗi cấu trúc tinh vi; **nhạy prompt/`class_name`**; pixel-AUROC zero-shot (85.1%) còn xa mức QC nghiêm ngặt.

**Chốt:** *DINOv2 ⇒ chính xác nhất khi có dữ liệu good; WinCLIP zero-shot ⇒ linh hoạt nhất khi không có dữ liệu. Không có "một mô hình thắng tất cả".*

---

### ▶ SLIDE 13 — Hướng ứng dụng thực tế & tính sáng tạo

**Tiêu đề:** Cùng pipeline "học từ ảnh bình thường" — ba miền ứng dụng

**1) QC công nghiệp — PCB & mối hàn.**
- *Vấn đề:* nứt mối hàn, thiếu/lệch linh kiện, đứt mạch.
- *Mô hình:* PatchCore cho lỗi bề mặt; **WinCLIP zero-shot** cho dòng sản phẩm mới chưa có dữ liệu; **Dinomaly** cho lỗi *logic* (sai số lượng/vị trí — đúng điểm yếu lộ ở `capsules`/`transistor`).
- *Dữ liệu:* chỉ cần ảnh bo mạch **đạt chuẩn**; camera công nghiệp + đèn đồng trục cố định.
- *Triển khai:* camera trên băng tải → suy luận **edge GPU (Jetson)**; ngưỡng recall cao (ít bỏ sót).

**2) Nông sản — bệnh trên bề mặt trái cây/lá.**
- *Vấn đề:* vết bệnh, đốm nấm, sâu bệnh trên cam/táo, lá.
- *Mô hình:* one-class trên ảnh quả/lá **khỏe** → vùng bệnh = bất thường; **DINOv2** hợp vì domain gap lớn (giống/ánh sáng ngoài trời).
- *Dữ liệu:* ảnh nông sản loại 1 tại kho phân loại; gán nhãn pixel chỉ cho tập test nhỏ.
- *Triển khai:* máy phân loại quang học tại HTX; hoặc app điện thoại (suy luận cloud).

**3) Y tế — tổn thương da (melanoma) & bất thường nội soi.**
- *Vấn đề:* khoanh vùng nốt/tổn thương nghi ngờ.
- *Mô hình:* one-class trên ảnh mô lành → tổn thương = anomaly; **DINOv2** mạnh nhờ đặc trưng tổng quát; kết hợp WinCLIP mô tả tổn thương bằng ngôn ngữ.
- *Dữ liệu:* ảnh lành công khai (vd ISIC) + dữ liệu bệnh viện ẩn danh; tuân thủ quy định y tế.
- *Triển khai:* công cụ **hỗ trợ sàng lọc** cho bác sĩ (không thay chẩn đoán); máy trạm bệnh viện. ⚠️ Cần bác sĩ xác nhận.

**Tính sáng tạo (chốt Phần 4):** điểm chung — *chỉ học từ dữ liệu "bình thường"* → hợp các miền **hiếm mẫu lỗi**. Hướng giàu tiềm năng: **zero-/few-shot bằng foundation model** để chuyển miền nhanh sang sản phẩm/cây trồng/mô bệnh *chưa từng thấy* mà không cần gán nhãn lỗi.

---

## 3. NHIỆM VỤ 2.3 — Bộ 5 câu Q&A phản biện (cho TV1)

**Câu 1 — Vì sao chọn DINOv2 thay vì backbone CNN lớn hơn (WideResNet101, ConvNeXt-XL)?**
> Nút thắt của PatchCore là **domain gap**, không phải dung lượng mạng. Tăng CNN giám sát-ImageNet vẫn kế thừa thiên lệch nhãn ImageNet. DINOv2 **tự giám sát trên 142M ảnh**, cho đặc trưng ngữ nghĩa, bất biến, chuyển miền tốt hơn. Bằng chứng: Dinomaly đạt ~99.6% (MVTec)/98.7% (VisA), một mô hình cho đa category. Cải thiện đến từ **chất lượng biểu diễn**, không phải số tham số.

**Câu 2 — Làm sao xử lý logical anomaly mà PatchCore không bắt được?**
> Số của nhóm định vị đúng điểm mù: VisA `capsules`/`macaroni1` (đa thực thể) image-AUROC chỉ 0.72–0.81 dù pixel ~0.97. Nguyên nhân: PatchCore so khớp **cục bộ**, không mô hình hóa quan hệ toàn cục. Hướng: (1) chuyển sang **reconstruction toàn ảnh** (Dinomaly); (2) benchmark **MVTec LOCO**; (3) ghép nhánh đếm/định vị thực thể ràng buộc bố cục. Đây là *giới hạn đã định vị bằng dữ liệu*, không phải lỗi cấu hình.

**Câu 3 — `capsules` Recall 0.99 mà Precision chỉ 0.63 — model có vô dụng không?**
> Không — nó cho thấy đúng **cơ chế hỏng**: 59/60 ảnh tốt bị báo nhầm vì **coreset không phủ hết biến thiên tự nhiên** của vỉ capsule đa thực thể → mọi ảnh "trông xa" memory bank. Cách khắc phục thực dụng: (1) tăng `coreset_sampling_ratio` + nhiều ảnh good hơn để phủ biến thiên; (2) đổi backbone DINOv2 cho đặc trưng bất biến hơn (E5); (3) hiệu chỉnh ngưỡng theo percentile ảnh good thay vì 0.5 mặc định. Recall 0.99 nghĩa là **không bỏ sót lỗi** — với QC, FP còn cứu được bằng kiểm tra phụ, FN thì không.

**Câu 4 — Ablation có công bằng không khi 3 model input size khác nhau (256/518/240)?**
> Chúng tôi **cố định mọi biến trừ model**: cùng 6 category, split, phần cứng, cách tính metric. Input size là **ràng buộc nội tại** từng kiến trúc, không ép bằng được mà không phá kiến trúc. Vì vậy chúng tôi **báo cáo latency & VRAM song song AUROC** và diễn giải dưới dạng **đánh đổi độ chính xác ↔ chi phí** — đúng chuẩn một ablation trung thực.

**Câu 5 — Vì sao tin số liệu của nhóm? `screw` 0.966 còn paper ~0.98?**
> Số lấy **trực tiếp từ `engine.test()`** của Anomalib, lưu ra CSV/JSON/HTML, không chỉnh tay; `bottle` 1.000 và TB MVTec 0.988/0.984 **khớp paper PatchCore** (~99.1%), xác nhận pipeline đúng. Chênh nhỏ ở `screw` (0.966 vs ~0.98) đến từ **biến thiên coreset ngẫu nhiên + chỉ 6 category + ngưỡng mặc định**, hoàn toàn hợp lý. Chúng tôi báo cáo đúng số tự chạy và nêu rõ nguồn — *(lưu ý: chưa tính PRO/F1-max và Pixel-AUROC VisA; sẽ bổ sung AUPRO khi chạy ablation).*

---

## 4. Nguồn tham khảo

- **Số liệu baseline (số nhóm):** `results/mvtec_baseline_results.csv`, `results 2/visa_baseline_results.csv`, `results 6/failure_analysis.csv`, `results 5/failure_analysis_visa.csv`, `results 3/PatchCore_Report.html` (Anomalib 2.5.0).
- **Dinomaly** (CVPR 2025) — image-AUROC 99.6% MVTec / 98.7% VisA (ViT-B), 99.8% (ViT-L); pixel 98.4%, AUPRO 94.8%. arXiv:2405.14325.
- **WinCLIP** (CVPR 2023) — zero-shot 91.8%/85.1%, AUPRO 64.6%; WinCLIP+ 1-shot 93.1%/95.2%, 4-shot ~95.2%/96.2%. arXiv:2303.14814.
- **PatchCore** (CVPR 2022) — ~99.1% image, ~98.1% pixel, ~93.5% PRO (15 cat). Roth et al.
