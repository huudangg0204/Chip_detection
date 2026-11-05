# Developer Specification: Chip Anomaly Detection System

> **Trạng thái:** Approved Specification (Sẵn sàng cho giai đoạn Implementation)  
> **Tài liệu gốc:** `docs/chip_anomaly_detection_project.md`  
> **Đối tượng áp dụng:** Team Dev (CV/Preprocessing, AI/Model Training, Backend/Pipeline, UI/Streamlit)

---

## 1. Objective & Confirmed Intent

### 1.1. Tóm tắt mục tiêu (Confirmed Intent)
- **Outcome (Sản phẩm đầu ra):** Hệ thống kiểm tra lỗi bo mạch điện tử (module cảm biến hồng ngoại LM393 trong `image/sample1.jpg` và `image/sample2.jpg`) theo hướng **Normal-only Anomaly Detection (PatchCore)**, bao gồm đầy đủ **Phase 1 (Upload ảnh tĩnh)** và **Phase 2 (Camera kiểm tra tự động/thủ công)** trên giao diện Streamlit.
- **User (Người dùng):**
  - **Team Dev:** Sử dụng tài liệu này làm chuẩn giao tiếp dữ liệu (Data Contract) để chia module code độc lập mà không bị xung đột khi ghép nối.
  - **Người vận hành / Giảng viên chấm đồ án:** Tương tác trực tiếp trên giao diện Streamlit để kiểm tra mạch thật, quan sát bản đồ nhiệt (Heatmap) vị trí lỗi và tinh chỉnh ngưỡng quyết định $\tau$ trực tiếp.
- **Why Now (Bối cảnh kỹ thuật thực tế):**
  - Qua khảo sát ảnh mẫu thực tế (`image/sample1.jpg`, `image/sample2.jpg`), bo mạch chỉ chiếm **<10% diện tích khung hình**, đặt lệch vị trí trên nền giấy trắng, trong khi lỗi thực tế (thiếu linh kiện dán SMD cạnh biến trở `103`) chỉ rộng vài chục pixel.
  - Nếu đưa nguyên ảnh gốc vào PatchCore, mô hình sẽ báo lỗi giả (False Positive) do dịch chuyển vị trí mạch và bóng đổ. Do đó, **Pipeline Tiền xử lý (Auto-Crop, Deskew, Định hướng 180° và Chuẩn hóa ánh sáng)** là thành phần bắt buộc.
- **Success Criteria (Tiêu chí thành công):**
  - Tự động tách nền, cắt sát (Crop) và xoay thẳng (Deskew) bo mạch từ ảnh chụp hoặc frame camera bất chấp lệch vị trí và thay đổi độ sáng nhẹ.
  - Phân biệt chính xác mạch `Normal` (`is_normal = True`) và mạch `Anomaly` (`is_normal = False`) với **AUROC $\ge 0.92$** trên tập test độc lập.
  - Heatmap hiển thị đúng vùng linh kiện bị thiếu/lệch (ví dụ: vùng linh kiện SMD sát biến trở `103`).
  - Hỗ trợ cả **Auto-Trigger** (tự chụp khi mạch đứng yên trong vùng ROI) và **Manual Capture** (nút chụp tay) kèm thanh trượt (Slider) chỉnh nóng ngưỡng $\tau$ ngay lúc demo.
- **Constraints (Ràng buộc):**
  - Mô hình: **PatchCore** (`anomalib` / `PyTorch`) với backbone chuẩn **`wide_resnet50_2`** (ưu tiên độ chính xác phát hiện chi tiết nhỏ).
  - Môi trường: Huấn luyện (Training) trên **Google Colab (GPU)** với bộ dữ liệu tự chụp (~30–50 ảnh Normal góc 90° top-down đa điều kiện sáng), xuất trọng số (`.ckpt` / `.pt`) về chạy Inference trên **Laptop cá nhân**.
- **Out of Scope (Ngoài phạm vi MVP):**
  - Không dùng YOLO để đếm/định danh tên từng linh kiện.
  - Không tối ưu rút gọn model cho thiết bị nhúng cấu hình thấp (Raspberry Pi / Jetson Nano).
  - Không hỗ trợ góc chụp nghiêng 3D tự do (yêu cầu góc chụp cố định ~90° từ trên xuống).

---

## 2. Tech Stack

| Thành phần | Công nghệ / Thư viện | Phiên bản khuyến nghị | Vai trò |
| :--- | :--- | :--- | :--- |
| **Ngôn ngữ** | Python | `3.10` – `3.11` | Ngôn ngữ phát triển chính |
| **Deep Learning** | PyTorch & Torchvision | `>= 2.1.0` | Trích xuất đặc trưng (`wide_resnet50_2`) & tính toán tensor |
| **Anomaly Detection** | Anomalib | `>= 1.1.0` | Huấn luyện & suy luận mô hình **PatchCore** |
| **Computer Vision** | OpenCV (`opencv-python`) | `>= 4.8.0` | Tách nền, tìm contour, xoay thẳng mạch, chuẩn hóa sáng CLAHE, đọc Camera |
| **Data & Array** | NumPy, Pillow, SciPy | `>= 1.24.0` | Xử lý ma trận ảnh và tính toán bản đồ nhiệt |
| **Web UI** | Streamlit | `>= 1.32.0` | Giao diện người dùng cho Phase 1, Phase 2 và công cụ thu thập dữ liệu |
| **Testing & Lint** | Pytest, Ruff | `>= 8.0.0` | Kiểm thử đơn vị (Unit Test) và chuẩn hóa code |
| **Training Cloud** | Google Colab (T4 GPU) | Notebook `.ipynb` | Môi trường huấn luyện PatchCore và tìm ngưỡng $\tau$ tối ưu |

---

## 3. Commands (Các lệnh thực thi chuẩn)

```bash
# 1. Khởi tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Kiểm tra bộ tiền xử lý ảnh (Auto-Crop & Alignment) trên 2 ảnh mẫu hiện có
python -m src.preprocessing.board_aligner --input image/sample1.jpg --output outputs/debug_sample1.jpg
python -m src.preprocessing.board_aligner --input image/sample2.jpg --output outputs/debug_sample2.jpg

# 3. Huấn luyện mô hình PatchCore (chạy trên Colab hoặc máy local có GPU)
python -m src.training.train_patchcore --config configs/patchcore_chip.yaml

# 4. Đánh giá mô hình và tìm ngưỡng quyết định (Threshold τ) tối ưu trên tập Validation/Test
python -m src.training.evaluate --config configs/patchcore_chip.yaml --weights weights/patchcore_best.ckpt

# 5. Chạy giao diện ứng dụng Streamlit (Phase 1 + Phase 2)
streamlit run src/ui/app.py

# 6. Chạy bộ kiểm thử tự động (Unit & Integration Tests)
pytest tests/ -v

# 7. Kiểm tra định dạng code (Linting)
ruff check src/ tests/
```

---

## 4. Project Structure (Cấu trúc thư mục dự án)

```text
Chip_detection/
├── configs/
│   └── patchcore_chip.yaml          # Cấu hình siêu tham số PatchCore, Preprocessing & Camera ROI
├── data/
│   └── chip_dataset/                # Bộ dữ liệu chuẩn hóa theo định dạng Folder của Anomalib
│       ├── normal/
│       │   ├── train/               # 30 - 50 ảnh Normal (đa điều kiện sáng, góc 90 độ)
│       │   └── val/                 # 10 - 15 ảnh Normal dùng để tính ngưỡng τ và kiểm thử
│       └── abnormal/
│           └── test/                # 10 - 15 ảnh Anomaly (thiếu/lệch/che linh kiện)
├── docs/
│   ├── chip_anomaly_detection_project.md  # Tài liệu tổng quan ý tưởng dự án
│   └── developer_spec.md                  # Tài liệu đặc tả kỹ thuật cho Team Dev (file này)
├── image/                           # Ảnh mẫu thô ban đầu (sample1.jpg, sample2.jpg)
├── notebooks/
│   └── Colab_Train_PatchCore.ipynb  # Notebook huấn luyện trên Google Colab & xuất file weights
├── outputs/                         # Thư mục lưu ảnh debug preprocessing và lịch sử kiểm tra
├── src/
│   ├── __init__.py
│   ├── schemas.py                   # Định nghĩa Dataclass chuẩn (InspectionResult, PreprocessResult)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── board_aligner.py         # Tách nền, tìm contour, xoay thẳng (Deskew) & định hướng 180°
│   │   └── illumination.py          # Cân bằng trắng Gray-World & chuẩn hóa sáng CLAHE (hệ màu LAB)
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── image_loader.py          # Đọc và kiểm tra định dạng ảnh tĩnh (Phase 1)
│   │   └── camera_trigger.py        # Quản lý Webcam & State Machine cho Auto-ROI Trigger (Phase 2)
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inspector.py             # Lớp bọc (Wrapper) InspectionModel kết nối Preprocessing + PatchCore
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_patchcore.py       # Script huấn luyện PatchCore qua Anomalib API
│   │   └── evaluate.py              # Script tính AUROC, F1-score và lưu ngưỡng τ tối ưu
│   └── ui/
│       └── app.py                   # Giao diện Streamlit tích hợp Phase 1, Phase 2 & Dataset Capture
├── tests/
│   ├── test_preprocessing.py        # Test cắt & xoay thẳng trên image/sample1.jpg và sample2.jpg
│   ├── test_camera_trigger.py       # Test logic chống chụp lặp (Debounce / State Machine)
│   └── test_inspector.py            # Test định dạng trả về của InspectionModel
├── weights/
│   ├── patchcore_best.ckpt          # Trọng số mô hình tải về từ Google Colab
│   └── metadata.json                # Lưu ngưỡng threshold τ mặc định và cấu hình normalize
└── requirements.txt
```

---

## 5. Đặc tả chi tiết từng Module (Core Technical Specifications)

### 5.1. Quy chuẩn Dataset & Chống nhiễu ánh sáng từ gốc (Data Protocol)

Để mô hình hoạt động chính xác kể cả khi đổi phòng demo (thay đổi ánh sáng), quy trình tạo dữ liệu phải tuân thủ:

1. **Số lượng ảnh tối thiểu:**
   - `data/chip_dataset/normal/train/`: **30 – 50 ảnh Normal** (chỉ chứa bo mạch lành lặn, đầy đủ linh kiện như `image/sample2.jpg`).
   - `data/chip_dataset/normal/val/`: **10 – 15 ảnh Normal** (chụp độc lập, không trùng tập train).
   - `data/chip_dataset/abnormal/test/`: **10 – 15 ảnh Anomaly** (mạch thiếu linh kiện SMD như `image/sample1.jpg`, hoặc tạo lỗi giả lập vật lý bằng cách dán mảnh băng dính nhỏ 1–2mm che lên trở dán, đặt lệch linh kiện, làm cong chân cắm).
2. **Quy tắc góc chụp (Camera Geometry):**
   - **Cố định góc 90° từ trên xuống (Top-down):** Không chụp góc nghiêng 3D.
   - **Biến thiên vị trí 2D hợp lệ:** Khi chụp 30–50 ảnh Normal, mỗi tấm xê dịch nhẹ vị trí mạch trên nền giấy trắng (vài cm) và xoay nhẹ $\pm 5^\circ \text{–} 15^\circ$ để sau khi qua bước OpenCV Auto-Crop, Memory Bank học được độ dung sai tự nhiên.
3. **Quy tắc Đa điều kiện sáng (Multi-Lighting Capture):**
   - Chia 40 ảnh Normal lúc chụp thành **4 nhóm ánh sáng** (mỗi nhóm ~10 ảnh):
     1. Ánh sáng đèn phòng tự nhiên.
     2. Bật đèn bàn chiếu xiên từ bên trái.
     3. Bật đèn bàn chiếu xiên từ bên phải / phía trên.
     4. Điều kiện ánh sáng hơi yếu hơn bình thường.
   - **Mục đích:** Giúp Memory Bank của PatchCore ghi nhận bóng đổ của biến trở `103` và vệt lóa mối hàn ở nhiều hướng khác nhau đều là trạng thái `Normal`.
4. **Data Augmentation lúc Train (trên Colab):**
   - Áp dụng `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)` và `GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))`.
   - **CẤM DÙNG:** `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCrop` (vì bo mạch có tính bất đối xứng trái-phải và trên-dưới; lật ảnh sẽ làm đảo vị trí linh kiện).

---

### 5.2. Module 1: Preprocessing Pipeline (`src/preprocessing/`)

Đây là module quan trọng nhất giúp biến ảnh chụp thô (như `image/sample1.jpg` kích thước lớn, mạch chiếm <10%) thành ảnh chuẩn hóa `256x256` chỉ chứa đúng bo mạch thẳng đứng.

```text
Raw Image (RGB)
   │
   ▼
[1. Board Segmentation] ──► Chuyển sang LAB/HSV + Otsu/Adaptive Threshold + Morphology Close
   │                        (Tách bo mạch tối màu khỏi nền giấy sáng bất chấp bóng đổ)
   ▼
[2. Contour & Deskew]   ──► Tìm Contour lớn nhất hợp lệ -> cv2.minAreaRect -> WarpAffine
   │                        (Cắt sát khung bo mạch + đệm 2% margin, dựng thẳng đứng H > W)
   ▼
[3. 180° Orientation]   ──► Kiểm tra đầu-cuối (2 LED ở trên, 3 chân kim loại G/V+/S ở dưới)
   │                        (Nếu bị ngược đầu 180° thì xoay lại cv2.ROTATE_180)
   ▼
[4. Light Normalization]──► Gray-World White Balance + CLAHE nhẹ trên kênh L (clipLimit=1.5)
   │
   ▼
[5. Resize & Pad]       ──► Resize giữ tỷ lệ hoặc chuẩn hóa về (256, 256) cho PatchCore
```

#### Các quy tắc thuật toán bắt buộc trong `board_aligner.py`:
1. **Tách nền chống bóng đổ:**
   - Chuyển ảnh sang hệ màu `LAB` hoặc `HSV`. Vì bo mạch có màu đen/xanh đen và nền giấy màu trắng sáng, sử dụng kênh `L` (Lightness) kết hợp `cv2.GaussianBlur` và ngưỡng tự động **Otsu (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`)**.
   - Dùng phép đóng hình thái học (`cv2.morphologyEx` với `MORPH_CLOSE`, kernel `7x7`) để lấp đầy các chi tiết sáng màu bên trên mặt mạch (như biến trở trắng/xanh `103`, lỗ ốc, chân hàn) thành một khối đặc duy nhất.
2. **Lọc Contour & Xoay thẳng (`Deskew`):**
   - Tìm contour có diện tích lớn nhất满足 điều kiện: chiếm từ `1%` đến `70%` diện tích ảnh và tỷ lệ khung hình (Aspect Ratio `max(w,h) / min(w,h)`) nằm trong khoảng `[2.0, 6.0]` (đặc trưng hình chữ nhật dài của module cảm biến).
   - Nếu không tìm thấy contour hợp lệ, trả về `PreprocessResult(success=False, error_message="Không tìm thấy bo mạch hợp lệ trong khung hình")` — **tuyệt đối không đưa ảnh nền trống vào PatchCore**.
   - Dùng `rect = cv2.minAreaRect(contour)` để lấy góc nghiêng, thực hiện `cv2.warpAffine` để xoay thẳng bo mạch sao cho chiều dài luôn nằm theo trục dọc (`Height > Width`), và cắt sát kèm lề đệm `margin = 2%`.
3. **Định hướng đúng chiều đầu-cuối (180° Canonicalization):**
   - Sau khi dựng dọc, mạch có thể bị lộn ngược 180°.
   - **Thuật toán kiểm tra chiều:** So sánh 25% vùng đỉnh (nơi có 2 bóng LED tròn và lỗ định vị) và 25% vùng đáy (nơi có 3 chân cắm kim loại sáng bóng + nền giấy trắng xen kẽ giữa 3 chân). Vùng đáy chứa 3 chân cắm trên nền trắng sẽ có mật độ pixel sáng / biên dọc cao hơn hẳn vùng đỉnh màu đen của cụm LED, hoặc dùng Template Matching nhẹ với vùng biến trở `103`. Nếu phát hiện ngược chiều, xoay `cv2.rotate(crop, cv2.ROTATE_180)`.
4. **Chuẩn hóa ánh sáng (`illumination.py`):**
   - Áp dụng **CLAHE** trên kênh `L` của không gian màu `LAB` với `clipLimit=1.5, tileGridSize=(8, 8)` để cân bằng độ sáng cục bộ mà không làm hỏng màu sắc linh kiện.

---

### 5.3. Module 2: PatchCore Training & Inference (`src/inference/inspector.py`)

#### Cấu hình mô hình chuẩn (`configs/patchcore_chip.yaml`):
```yaml
model:
  name: patchcore
  backbone: wide_resnet50_2
  pre_trained: true
  layers:
    - layer2
    - layer3
  coreset_sampling_ratio: 0.1
  num_neighbors: 9

preprocess:
  image_size: [256, 256]
  apply_clahe: true
  clahe_clip_limit: 1.5
  crop_margin_ratio: 0.02

decision:
  default_threshold: 0.50  # Sẽ được ghi đè tự động bằng giá trị τ tối ưu trong weights/metadata.json
```

#### Giao diện dữ liệu chuẩn (Data Contract trong `src/schemas.py`):
Mọi module (Phase 1, Phase 2, UI, Test) bắt buộc giao tiếp qua cấu trúc dữ liệu này:

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class PreprocessResult:
    success: bool
    cropped_rgb: Optional[np.ndarray] = None       # Ảnh mạch đã crop & xoay thẳng (H, W, 3)
    normalized_rgb: Optional[np.ndarray] = None    # Ảnh (256, 256, 3) sau CLAHE đưa vào model
    bbox_box_points: Optional[np.ndarray] = None   # 4 đỉnh của minAreaRect trên ảnh gốc để vẽ UI
    rotation_angle: float = 0.0
    error_message: Optional[str] = None

@dataclass
class InspectionResult:
    status: str                                    # "OK" hoặc "NO_BOARD_DETECTED" hoặc "ERROR"
    is_normal: Optional[bool] = None               # True: Normal (Không lỗi), False: Anomaly (Có lỗi)
    anomaly_score: Optional[float] = None          # Điểm bất thường từ PatchCore (đã chuẩn hóa hoặc raw)
    threshold: float = 0.50                        # Ngưỡng τ dùng tại thời điểm phán đoán
    cropped_board_rgb: Optional[np.ndarray] = None # Ảnh bo mạch đã cắt thẳng
    anomaly_map: Optional[np.ndarray] = None       # Ma trận nhiệt (256, 256) giá trị float
    heatmap_overlay_rgb: Optional[np.ndarray] = None # Ảnh bo mạch đè lớp màu Heatmap (Jet colormap)
    processing_time_ms: float = 0.0                # Tổng thời gian xử lý (Preprocess + Inference)
    message: str = ""
```

---

### 5.4. Module 3: Camera Acquisition & Auto-Trigger (`src/acquisition/camera_trigger.py`)

Để tránh việc hệ thống chụp khi tay người dùng đang di chuyển mạch (gây nhòe ảnh) hoặc chụp lặp đi lặp lại cùng một mạch hàng chục lần mỗi giây, Phase 2 sử dụng **State Machine 4 trạng thái**:

```text
[1. IDLE_EMPTY] ──(Phát hiện mạch trong ROI)──► [2. STABILIZING]
       ▲                                                │
       │                                    (Đứng yên N=10 frames liên tiếp,
       │                                     độ lệch frame-to-frame < ε)
       │                                                ▼
[4. COOLDOWN_WAIT_EXIT] ◄──(Chạy PatchCore xong)── [3. TRIGGER_CAPTURE]
   (Chờ lấy mạch ra khỏi ROI)
```

1. **`IDLE_EMPTY`:** Khung ROI trống (diện tích contour trong ROI `< min_area`).
2. **`STABILIZING`:** Phát hiện có mạch trong ROI. Tính độ dịch chuyển tâm `(cx, cy)` và độ chênh lệch ảnh giữa frame hiện tại và frame trước đó. Khi mạch đứng yên liên tiếp `N = 10` frames (~0.35 giây ở 30 FPS), chuyển sang `TRIGGER_CAPTURE`.
3. **`TRIGGER_CAPTURE`:** Lấy frame nét nhất hiện tại đưa vào `InspectionModel.predict(frame, threshold=current_tau)` và cập nhật kết quả lên UI.
4. **`COOLDOWN_WAIT_EXIT`:** Khóa không chụp tiếp cho đến khi người dùng nhấc mạch hiện tại ra khỏi vùng ROI (diện tích contour về `0` trong ít nhất `5` frames) hoặc người dùng bấm nút **"Inspect Again / Manual Capture"**.

---

### 5.5. Module 4: Giao diện Streamlit (`src/ui/app.py`)

Giao diện được chia làm **Sidebar điều khiển chung** và **3 Tabs chức năng**:

#### Sidebar (Bảng điều khiển & Tinh chỉnh nóng khi Demo):
- **Model Status:** Hiển thị trạng thái đã load file trọng số (`weights/patchcore_best.ckpt`) và thiết bị chạy (`CUDA` / `CPU`).
- **Threshold Calibration Slider ($\tau$):** Thanh trượt `st.slider("Ngưỡng quyết định (Threshold τ)", 0.0, 1.0, default_tau, step=0.01)`.
  - *Đặc điểm quan trọng:* Khi kéo thanh trượt $\tau$, kết quả `TRUE (Normal)` / `FALSE (Anomaly)` của bức ảnh vừa kiểm tra sẽ **tự động cập nhật ngay lập tức** dựa trên công thức `score <= τ` mà không cần chạy lại model inference.
- **Preprocessing Options:** Checkbox bật/tắt CLAHE chuẩn hóa sáng và điều chỉnh độ nhạy tách nền nếu đổi môi trường phòng thi.

#### Tab 1: Phase 1 – Image-based Inspection (Kiểm tra qua ảnh tải lên)
- Khu vực tải ảnh (`st.file_uploader` hỗ trợ `.jpg`, `.jpeg`, `.png`) + nút chọn nhanh 2 ảnh mẫu có sẵn (`sample1.jpg`, `sample2.jpg`).
- Hiển thị trực quan 3 cột song song:
  1. **Ảnh gốc (Raw Image):** Có vẽ khung viền xanh bao quanh vị trí bo mạch được phát hiện.
  2. **Ảnh sau Preprocessing (Cropped & Aligned ROI):** Ảnh bo mạch đã cắt sát, xoay thẳng đứng và cân bằng sáng.
  3. **Bản đồ nhiệt (Anomaly Heatmap Overlay):** Vùng nghi ngờ lỗi tô màu đỏ/vàng rực trên nền bo mạch.
- Bảng kết quả nổi bật:
  - Badge lớn: **`TRUE — NORMAL (BÌNH THƯỜNG)`** (Màu xanh lá) hoặc **`FALSE — ANOMALY (PHÁT HIỆN LỖI)`** (Màu đỏ).
  - Các chỉ số: `Anomaly Score`, `Threshold (τ)`, `Processing Time (ms)`.

#### Tab 2: Phase 2 – Camera-based Inspection (Kiểm tra qua Camera thời gian thực)
- Chọn nguồn Camera (`Camera Index: 0, 1...`) và chế độ chụp:
  - **Chế độ 1: Auto-Trigger (Tự động chụp khi mạch ổn định trong ROI).**
  - **Chế độ 2: Manual Capture (Nút bấm chụp thủ công — dự phòng an toàn 100% khi bảo vệ đồ án).**
- Khung Live Video hiển thị:
  - Khung chữ nhật định vị ROI ở giữa màn hình.
  - Trạng thái hiện tại của State Machine (`Đang chờ đặt mạch...` $\rightarrow$ `Đang lấy nét ổn định (7/10)...` $\rightarrow$ `Đã chụp & Kiểm tra! Nhấc mạch ra để tiếp tục`).
- Khung kết quả bên phải hiển thị kết quả của lần chụp gần nhất + bảng lịch sử các lần kiểm tra trong phiên (`Timestamp`, `Score`, `Threshold`, `Result`).

#### Tab 3: Dataset Helper (Công cụ hỗ trợ chụp Dataset nhanh cho Team)
- Cho phép mở Camera, xem trước ảnh đã qua `board_aligner` (để chắc chắn mạch được cắt đẹp) và bấm nút lưu nhanh vào `data/chip_dataset/normal/train`, `normal/val`, hoặc `abnormal/test`.

---

## 6. Code Style & Reference Implementation

- **Quy chuẩn code:** Python 3.10+, bắt buộc dùng **Type Hints** cho mọi hàm, dùng `dataclass` cho dữ liệu trả về, dùng `logging` thay vì `print()` rải rác, và viết docstring ngắn gọn.
- **Mẫu code chuẩn cho `InspectionModel` (`src/inference/inspector.py`):**

```python
import time
from pathlib import Path
import numpy as np
from src.schemas import InspectionResult
from src.preprocessing.board_aligner import BoardAligner

class InspectionModel:
    """Đóng gói toàn bộ luồng Preprocessing + PatchCore Inference cho cả Phase 1 & Phase 2."""

    def __init__(self, weights_path: Path, default_threshold: float = 0.50) -> None:
        self.weights_path = weights_path
        self.default_threshold = default_threshold
        self.aligner = BoardAligner(target_size=(256, 256), apply_clahe=True)
        self._model = None

    def load(self) -> None:
        """Load trọng số PatchCore đã huấn luyện."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số model tại: {self.weights_path}")
        # Khởi tạo Anomalib TorchInferencer / Lightning model tại đây
        ...

    def predict(self, image_rgb: np.ndarray, threshold: float | None = None) -> InspectionResult:
        """Thực hiện tiền xử lý và dự đoán bất thường trên 1 ảnh RGB."""
        start_time = time.perf_counter()
        tau = threshold if threshold is not None else self.default_threshold

        # Bước 1: Tách nền, cắt sát và xoay thẳng bo mạch
        prep = self.aligner.process(image_rgb)
        if not prep.success or prep.normalized_rgb is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return InspectionResult(
                status="NO_BOARD_DETECTED",
                threshold=tau,
                processing_time_ms=round(elapsed_ms, 2),
                message=prep.error_message or "Không phát hiện được bo mạch trong ảnh.",
            )

        # Bước 2: Chạy PatchCore Inference trên ảnh bo mạch đã chuẩn hóa
        score, anomaly_map, heatmap_overlay = self._run_patchcore(prep.normalized_rgb)
        is_normal = bool(score <= tau)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return InspectionResult(
            status="OK",
            is_normal=is_normal,
            anomaly_score=round(float(score), 4),
            threshold=tau,
            cropped_board_rgb=prep.cropped_rgb,
            anomaly_map=anomaly_map,
            heatmap_overlay_rgb=heatmap_overlay,
            processing_time_ms=round(elapsed_ms, 2),
            message="Normal (Không phát hiện bất thường)" if is_normal else "Anomaly (Phát hiện bất thường)",
        )
```

---

## 7. Testing Strategy & Phân công công việc cho Team Dev

### 7.1. Chiến lược kiểm thử (`tests/`)
1. **Unit Test cho Preprocessing (`tests/test_preprocessing.py`):**
   - Chạy trực tiếp trên 2 ảnh thật `image/sample1.jpg` và `image/sample2.jpg`.
   - Kiểm tra: `prep.success == True`, ảnh đầu ra có kích thước `(256, 256, 3)`, bo mạch được dựng thẳng đứng (`cropped_rgb.shape[0] > cropped_rgb.shape[1]`), và 2 ảnh sau khi căn chỉnh có độ tương đồng cấu trúc (SSIM) cao ở vùng không lỗi.
   - Kiểm tra với 1 ảnh trắng trơn (không có mạch): phải trả về `prep.success == False`.
2. **Unit Test cho Camera Trigger (`tests/test_camera_trigger.py`):**
   - Giả lập chuỗi 15 frames đứng yên $\rightarrow$ xác nhận chỉ bắn ra đúng **1 lần trigger duy nhất** và chuyển sang trạng thái `COOLDOWN_WAIT_EXIT`.
3. **Contract Test cho Model Wrapper (`tests/test_inspector.py`):**
   - Kiểm tra với cùng một `anomaly_score = 0.62`: nếu truyền `threshold = 0.70` thì `is_normal == True`; nếu truyền `threshold = 0.50` thì `is_normal == False`.

### 7.2. Gợi ý phân chia công việc cho Team Dev (Parallel Workstreams)
Nhờ kiến trúc tách biệt qua `src/schemas.py`, các thành viên có thể code song song ngay từ ngày đầu tiên:

| Vai trò | Module phụ trách | Đầu việc cụ thể |
| :--- | :--- | :--- |
| **Dev 1 (CV & Preprocessing)** | `src/preprocessing/`, `src/acquisition/camera_trigger.py` | Viết thuật toán tách nền Otsu/LAB, xoay thẳng mạch `minAreaRect`, chống ngược 180°, cân bằng sáng CLAHE và bộ kích hoạt Camera tự động. |
| **Dev 2 (AI & Colab Training)** | `notebooks/`, `src/training/`, `src/inference/inspector.py` | Chụp bộ dataset (30–50 Normal + 15 Test), xây dựng Colab Notebook train PatchCore (`wide_resnet50_2`), tìm ngưỡng $\tau$ và viết hàm load checkpoint. |
| **Dev 3 (UI & Integration)** | `src/ui/app.py`, `src/schemas.py` | Xây dựng giao diện Streamlit (Tab Phase 1, Tab Phase 2, Tab chụp Dataset, thanh trượt chỉnh nóng $\tau$) sử dụng Mock data trước khi ghép model thật. |

---

## 8. Boundaries (Nguyên tắc bắt buộc cho Team Dev)

- **Always Do (Luôn phải làm):**
  - Luôn chạy ảnh qua `BoardAligner` trước khi đưa vào `PatchCore` (cả lúc Train lẫn lúc Inference phải dùng chung 1 hàm tiền xử lý y hệt nhau).
  - Luôn giữ định dạng màu thống nhất là **RGB** ở các giao diện hàm (`OpenCV` mặc định đọc `BGR` nên phải `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` ngay tại tầng đọc ảnh/camera).
  - Luôn chạy `pytest tests/` trước khi tạo Pull Request / gộp code.
- **Ask First (Hỏi cả nhóm trước khi thay đổi):**
  - Thay đổi kích thước ảnh đầu vào chuẩn `(256, 256)` hoặc đổi tên các trường trong `src/schemas.py`.
  - Thêm thư viện nặng mới vào `requirements.txt`.
- **Never Do (Tuyệt đối không làm):**
  - **Không bao giờ** đưa ảnh `Abnormal/Anomaly` vào thư mục `normal/train/` của PatchCore.
  - **Không bao giờ** dùng `RandomHorizontalFlip` hoặc `RandomVerticalFlip` khi train bo mạch này.
  - **Không bao giờ** commit các file trọng số nặng hàng trăm MB (`.ckpt`, `.pt`) hoặc toàn bộ thư mục video/ảnh thô nặng trực tiếp lên Git (dùng `.gitignore` và lưu qua Google Drive / script tải tự động).

---

## 9. Acceptance Checklist (Tiêu chí nghiệm thu toàn dự án)

- [ ] **Tiền xử lý:** Chạy lệnh test trên `image/sample1.jpg` và `image/sample2.jpg` cắt ra đúng 2 ảnh bo mạch sát viền, thẳng đứng, cùng chiều và đồng đều độ sáng.
- [ ] **Dataset & Huấn luyện:** Có bộ dataset chuẩn ($\ge 30$ ảnh Normal đa góc sáng + $\ge 15$ ảnh Val/Test) và Notebook Colab chạy ra được file trọng số + ngưỡng $\tau$ với AUROC $\ge 0.92$.
- [ ] **Phase 1 (Upload UI):** Tải ảnh Normal trả về `TRUE` (Xanh), tải ảnh thiếu linh kiện SMD (`sample1.jpg`) trả về `FALSE` (Đỏ) kèm Heatmap sáng đỏ ngay tại vị trí linh kiện bị thiếu cạnh biến trở `103`.
- [ ] **Phase 2 (Camera UI):** Đặt mạch thật vào khung ROI dưới Webcam $\rightarrow$ tự động nhận diện khi mạch đứng yên (hoặc bấm chụp tay), trả về kết quả `TRUE/FALSE` chính xác và không bị chụp lặp liên tục khi chưa lấy mạch ra.
- [ ] **Chống nhiễu môi trường:** Có thanh trượt chỉnh nóng ngưỡng $\tau$ trên giao diện Streamlit để thích ứng tức thì tại phòng bảo vệ đồ án.
