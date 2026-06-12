### Đề tài 1: Visual Anomaly Detection (VAND Challenge – nền tảng MVTec AD / VisA)

**📌 Thông tin cơ bản**

|Mục|Chi tiết|
|---|---|
|Tên đầy đủ|VAND – Visual Anomaly and Novelty Detection Challenge (trên dataset MVTec AD / VisA)|
|Hội nghị/Nguồn|CVPR 2023 & CVPR 2024 Workshop "VAND" **[link challenge cần xác minh]**|
|Năm|2023–2024 (dataset MVTec AD: 2019, vẫn là benchmark chuẩn)|
|Nhóm bài toán|Anomaly Detection|
|Link chính thức|Workshop: tìm "VAND CVPR workshop" **[cần xác minh]**; dataset MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad|

**🔬 Mô tả kỹ thuật**

- **Bài toán cụ thể:** Phát hiện và định vị bất thường (defect) trên ảnh sản phẩm công nghiệp. Input: ảnh sản phẩm; Output: (1) điểm bất thường mức ảnh (image-level score) và (2) bản đồ nhiệt định vị vùng lỗi mức pixel (pixel-level localization). Đặc thù: chỉ huấn luyện trên ảnh "bình thường", không có/rất ít mẫu lỗi (unsupervised / few-shot).
- **Dataset:** MVTec AD — ~5.354 ảnh độ phân giải cao, 15 nhóm (10 object + 5 texture). VAND 2024 mở rộng sang track zero-shot/few-shot trên VisA, MVTec LOCO **[số ảnh cụ thể cần xác minh]**.
- **Metric đánh giá:** Image-level AUROC, Pixel-level AUROC, PRO (Per-Region Overlap), F1-max.
- **Baseline/SOTA:** PatchCore (~99% image AUROC trên MVTec AD); các phương pháp mới (~99.5%+) **[con số cần xác minh]**. Track zero-shot (dùng CLIP/foundation model như WinCLIP) chưa bão hòa, AUROC thấp hơn nhiều.

**💻 Code & Reproducibility**

- **GitHub repo:** https://github.com/open-edge-platform/anomalib (thư viện chính thức, gồm PatchCore, PaDiM, FastFlow, EfficientAD…). Repo phụ: https://github.com/amazon-science/patchcore-inspection
- **Stars/Forks:** anomalib ~5.828 sao / ~938 fork; patchcore-inspection ~1.300 sao (xác minh 10/06/2026)
- **Ngôn ngữ/Framework:** Python, PyTorch + PyTorch Lightning
- **Pretrained weights:** Có — backbone ImageNet sẵn; PatchCore không cần train (memory bank), chỉ cần fit + inference
- **Ước tính setup:** 1–2 giờ. GPU 8GB đủ; PatchCore/PaDiM chạy được cả trên Colab free
- **Colab/Notebook:** Có — anomalib có thư mục `examples/notebooks` (Jupyter) hướng dẫn từng bước

**🌍 Ứng dụng thực tiễn & Tính sáng tạo**

- **Ứng dụng chính:** Kiểm tra chất lượng (QC) trong sản xuất (PCB, dệt may, dược phẩm); phát hiện lỗi bề mặt trên dây chuyền; kiểm định linh kiện.
- **Ý tưởng mở rộng sáng tạo:** (1) Few-shot/zero-shot defect detection dùng foundation model (CLIP/SAM/DINOv2) để phát hiện lỗi loại sản phẩm chưa từng thấy — rất nóng và chưa bão hòa. (2) Áp dụng sang lĩnh vực mới ngoài công nghiệp: phát hiện bất thường ảnh nông sản (vết bệnh trên trái cây) hoặc ảnh y tế (tổn thương da) bằng cùng pipeline.
- **Đối tượng hưởng lợi:** Nhà máy/SME sản xuất, kỹ sư QC, ngành kiểm định.

**📊 Phân tích sâu**

- **Thách thức kỹ thuật chính:** (1) Học chỉ từ dữ liệu bình thường (không nhãn lỗi); (2) định vị pixel chính xác cho lỗi nhỏ; (3) cân bằng giữa độ chính xác và tốc độ cho edge deployment.
- **Điểm mạnh của đề tài:** Khả thi cao nhất — PatchCore không cần train, chạy được trên Colab, có pretrained và notebook. Leaderboard công khai trên Papers with Code để so sánh.
- **Điểm yếu/Rủi ro:** MVTec AD "cổ điển" đã gần bão hòa (>99%), nên cần chọn track khó (zero/few-shot, MVTec LOCO logical anomaly) để thể hiện đóng góp.
- **Hướng cải thiện khả thi:** (1) Thay backbone bằng DINOv2 và đo độ cải thiện; (2) thử WinCLIP cho zero-shot rồi phân tích vì sao một số class thất bại.

**⭐ Đánh giá tổng quan**

|Tiêu chí|Điểm (1–5)|
|---|---|
|Độ khả thi|5/5|
|Tính ứng dụng & sáng tạo|5/5|
|Độ sâu kỹ thuật|5/5|
|Tính mới|4/5|
|**Tổng**|**19/20**|

---

### Đề tài 2: Agriculture-Vision Prize Challenge (CVPR Workshop)

**📌 Thông tin cơ bản**

|Mục|Chi tiết|
|---|---|
|Tên đầy đủ|Agriculture-Vision Prize Challenge & Workshop|
|Hội nghị/Nguồn|CVPR Workshop (các kỳ 2020, 2021, 2022, 2023) — thuộc khoảng 2022–2023 hợp lệ|
|Năm|2022–2023 (dataset gốc 2020, mở rộng các năm sau)|
|Nhóm bài toán|Segmentation (semantic segmentation đa nhãn)|
|Link chính thức|https://www.agriculture-vision.com/ ; dataset HuggingFace: https://huggingface.co/datasets/shi-labs/Agriculture-Vision|

**🔬 Mô tả kỹ thuật**

- **Bài toán cụ thể:** Phân vùng ngữ nghĩa các "bất thường nông nghiệp" trên ảnh không ảnh (aerial) ruộng đồng. Input: ảnh 512×512 4 kênh (RGB + NIR) kèm boundary/mask; Output: mask pixel cho từng loại hiện tượng (double plant, nutrient deficiency, weed cluster, water, waterway…).
- **Dataset:** Challenge dataset ~21.061 ảnh 512×512, chụp năm 2019 khắp Hoa Kỳ; 6 pattern + background ở kỳ đầu, mở rộng tới ~9 loại ở các kỳ sau **[số class theo từng kỳ cần xác minh]**. File ~20 GB.
- **Metric đánh giá:** mIoU có biến thể (cho phép nhãn chồng lấn — pixel có nhiều nhãn, dự đoán trúng một nhãn vẫn tính đúng).
- **Baseline/SOTA:** DeepLabV3+/HRNet/U-Net là baseline; SOTA mIoU trên leaderboard Codalab **[con số cần xác minh]**.

**💻 Code & Reproducibility**

- **GitHub repo:** https://github.com/SHI-Labs/Agriculture-Vision (mô tả dataset, metric, link leaderboard, baseline)
- **Stars/Forks:** ~257 sao / ~40 fork (xác minh 10/06/2026)
- **Ngôn ngữ/Framework:** Python, PyTorch (baseline thường dựa trên mmsegmentation)
- **Pretrained weights:** Backbone ImageNet sẵn; trọng số challenge cụ thể **[cần xác minh trên repo/leaderboard]**
- **Ước tính setup:** 3–5 giờ (gồm tải ~20 GB). GPU 12–16GB train được DeepLab/HRNet với batch nhỏ
- **Colab/Notebook:** Không có notebook chính thức nổi bật **[cần xác minh]** — nhưng pipeline mmsegmentation rất chuẩn hóa

**🌍 Ứng dụng thực tiễn & Tính sáng tạo**

- **Ứng dụng chính:** Nông nghiệp chính xác (precision agriculture) — phát hiện vùng thiếu dinh dưỡng, cỏ dại, úng nước để bón phân/tưới tiêu đúng chỗ; giám sát sức khỏe cây trồng bằng drone/vệ tinh.
- **Ý tưởng mở rộng sáng tạo:** (1) Khai thác kênh NIR + chỉ số NDVI làm đầu vào bổ sung và đo mức cải thiện so với chỉ RGB; (2) chuyển giao mô hình sang ảnh drone của ruộng lúa Việt Nam (domain adaptation) — hướng địa phương hóa hiếm người làm.
- **Đối tượng hưởng lợi:** Nông dân, hợp tác xã, công ty agritech, cơ quan quản lý đất nông nghiệp.

**📊 Phân tích sâu**

- **Thách thức kỹ thuật chính:** (1) Nhãn chồng lấn và mất cân bằng lớp nghiêm trọng; (2) tận dụng kênh NIR (4 kênh thay vì 3); (3) đối tượng nhỏ, ranh giới mờ trên ảnh không ảnh.
- **Điểm mạnh của đề tài:** Là challenge CVPR thực thụ, có leaderboard Codalab công khai, ứng dụng nông nghiệp rõ ràng và dễ "địa phương hóa" cho đồ án.
- **Điểm yếu/Rủi ro:** Dataset 20GB; xử lý ảnh 4 kênh + nhãn chồng lấn đòi hỏi sửa data loader; cần train (tốn GPU hơn Đề tài 1).
- **Hướng cải thiện khả thi:** (1) Class-balanced loss / focal loss cho lớp hiếm; (2) thêm NDVI làm kênh thứ 5 và phân tích ablation.

**⭐ Đánh giá tổng quan**

|Tiêu chí|Điểm (1–5)|
|---|---|
|Độ khả thi|4/5|
|Tính ứng dụng & sáng tạo|5/5|
|Độ sâu kỹ thuật|4/5|
|Tính mới|4/5|
|**Tổng**|**17/20**|

---

### Đề tài 3: iWildCam Challenge (FGVC @ CVPR) – phân loại loài từ camera trap

**📌 Thông tin cơ bản**

|Mục|Chi tiết|
|---|---|
|Tên đầy đủ|iWildCam Challenge (FGVC9 Workshop, CVPR 2022) — phần phân loại loài thuộc benchmark WILDS|
|Hội nghị/Nguồn|CVPR 2022, FGVC9 Workshop (kỳ 2022 tập trung đếm cá thể; phân loại loài dùng phiên bản iWildCam-WILDS)|
|Năm|2022 (challenge); WILDS được duy trì liên tục|
|Nhóm bài toán|Classification (long-tailed + domain shift)|
|Link chính thức|https://github.com/visipedia/iwildcam_comp ; WILDS: https://wilds.stanford.edu/|

**🔬 Mô tả kỹ thuật**

- **Bài toán cụ thể:** Phân loại loài động vật trong ảnh camera trap. Điểm đặc biệt: ảnh train và test đến từ các camera/địa điểm khác nhau (distribution shift) — mô hình phải tổng quát hóa sang camera chưa từng thấy. (Kỳ CVPR 2022 chuyển sang bài toán đếm cá thể qua chuỗi ảnh.)
- **Dataset:** iWildCam-WILDS ~203.000 ảnh, ~182 lớp loài, chia theo camera **[con số cần xác minh]**. Kèm MegaDetector để crop con vật.
- **Metric đánh giá:** Macro-F1 (vì mất cân bằng lớp nặng); Accuracy.
- **Baseline/SOTA:** ResNet-50 (ERM) là baseline trong WILDS; các phương pháp domain generalization (IRM, CORAL…) trên leaderboard WILDS **[con số cần xác minh]**.

**💻 Code & Reproducibility**

- **GitHub repo:** https://github.com/p-lambda/wilds (data loader + model + evaluator chuẩn) và https://github.com/visipedia/iwildcam_comp (mô tả challenge)
- **Stars/Forks:** WILDS ~600 sao / ~135 fork; iwildcam_comp ~151 sao / ~21 fork (xác minh 10/06/2026)
- **Ngôn ngữ/Framework:** Python, PyTorch
- **Pretrained weights:** Có — WILDS cung cấp checkpoint ResNet-50 đã train sẵn cho iWildCam
- **Ước tính setup:** 3–6 giờ (tải dữ liệu lớn ~40GB). GPU 12–16GB chạy được; có thể dùng subset cho Colab
- **Colab/Notebook:** WILDS có ví dụ chạy mẫu/README rõ ràng; notebook chính thức **[cần xác minh]**

**🌍 Ứng dụng thực tiễn & Tính sáng tạo**

- **Ứng dụng chính:** Bảo tồn đa dạng sinh học — tự động kiểm kê loài, ước lượng mật độ quần thể từ hàng triệu ảnh bẫy ảnh; cảnh báo loài xâm lấn.
- **Ý tưởng mở rộng sáng tạo:** (1) Test domain generalization bằng dữ liệu camera trap Việt Nam (vườn quốc gia) — đo độ rớt accuracy khi đổi vùng; (2) kết hợp MegaDetector + classifier thành pipeline đếm cá thể nhẹ chạy gần thời gian thực.
- **Đối tượng hưởng lợi:** Nhà sinh thái học, vườn quốc gia/khu bảo tồn, tổ chức bảo tồn (WCS).

**📊 Phân tích sâu**

- **Thách thức kỹ thuật chính:** (1) Distribution shift giữa các camera; (2) phân bố lớp long-tailed cực đoan; (3) ảnh nhiễu (đêm, nhòe, con vật một phần khung hình).
- **Điểm mạnh của đề tài:** Góc nghiên cứu "distribution shift" rất sâu, lý tưởng để thể hiện hiểu biết; có leaderboard WILDS công khai và pretrained.
- **Điểm yếu/Rủi ro:** Dữ liệu lớn (~40GB); accuracy out-of-distribution thấp dễ gây nản; phần đếm cá thể (kỳ 2022) khó hơn phân loại thuần.
- **Hướng cải thiện khả thi:** (1) Thử các thuật toán domain generalization (CORAL, GroupDRO) và so với ERM; (2) test-time adaptation rồi phân tích cải thiện trên camera mới.

**⭐ Đánh giá tổng quan**

|Tiêu chí|Điểm (1–5)|
|---|---|
|Độ khả thi|3/5|
|Tính ứng dụng & sáng tạo|4/5|
|Độ sâu kỹ thuật|5/5|
|Tính mới|4/5|
|**Tổng**|**16/20**|

---

### 🏆 Khuyến nghị cuối cùng

- **Đề tài được khuyến nghị:** Đề tài 1 — Visual Anomaly Detection (anomalib / VAND).
- **Lý do chọn:** Đây là lựa chọn cân bằng tốt nhất cho một môn học: code chạy được gần như chắc chắn (PatchCore không cần train, có pretrained và Jupyter notebook, chạy được trên Colab free hoặc GPU 8GB), nhưng vẫn có chiều sâu để thể hiện hiểu biết qua track zero-shot/few-shot dùng foundation model — vốn chưa bão hòa. Ứng dụng QC công nghiệp rõ ràng và dễ mở rộng sáng tạo sang y tế/nông nghiệp, đồng thời có leaderboard công khai (Papers with Code) để đối chiếu kết quả. Đề tài 2 và 3 đáng cân nhắc nếu bạn muốn ứng dụng nông nghiệp/bảo tồn cụ thể, nhưng đòi hỏi tải dữ liệu lớn và train nhiều hơn.
- **Bước tiếp theo:**
    1. Clone `open-edge-platform/anomalib`, cài theo README, chạy notebook trong `examples/notebooks` với PatchCore trên 1–2 nhóm của MVTec AD để có kết quả baseline.
    2. Xác minh trang challenge VAND của CVPR 2023/2024 và tải đúng track (chuẩn / zero-shot / few-shot) cùng dataset tương ứng (MVTec AD, VisA hoặc MVTec LOCO).
    3. Chọn 1 hướng mở rộng (đổi backbone sang DINOv2, hoặc thử WinCLIP zero-shot), chạy ablation và ghi lại metric (image/pixel AUROC, PRO) để so với baseline trên leaderboard.

Lưu ý: các trang challenge của hội nghị và một số con số metric SOTA tôi chưa truy cập trực tiếp được nên đã đánh dấu **[cần xác minh]** — bạn nên kiểm tra lại trên trang workshop chính thức trước khi đưa vào báo cáo. Nếu muốn, tôi có thể chạy thử pipeline anomalib và dựng một notebook baseline mẫu cho bạn.