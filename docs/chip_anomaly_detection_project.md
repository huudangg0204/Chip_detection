# Chip Anomaly Detection System

## 1. Tổng quan dự án

**Chip Anomaly Detection System** là hệ thống Computer Vision ứng dụng Deep Learning để kiểm tra tình trạng của chip thông qua hình ảnh. Hệ thống học đặc trưng từ hình ảnh chip bình thường (*Normal*) và sử dụng các đặc trưng đó để phát hiện hình ảnh có dấu hiệu bất thường (*Anomaly*).

Dự án được chia thành hai giai đoạn phát triển:

- **Phase 1 – Image-based Inspection:** Người dùng tải một ảnh chip lên hệ thống. Hệ thống phân tích ảnh và trả về kết quả Boolean:
  - `true`: Không phát hiện anomaly (Normal).
  - `false`: Phát hiện anomaly.
- **Phase 2 – Camera-based Inspection:** Camera tự động thu nhận hình ảnh chip khi chip xuất hiện trong vùng kiểm tra. Hệ thống sử dụng cùng pipeline AI của Phase 1 để phân tích và trả về `true` hoặc `false`.

Mục tiêu là xây dựng một prototype gọn, có thể trình diễn trong phạm vi đồ án môn học, đồng thời giữ kiến trúc đủ linh hoạt để mở rộng từ xử lý ảnh tĩnh sang kiểm tra bằng camera.

---

## 2. Bài toán

### 2.1. Định nghĩa

Bài toán được tiếp cận theo hướng **Normal-only / One-class Anomaly Detection**.

Thay vì huấn luyện mô hình để phân loại đầy đủ từng loại lỗi, hệ thống ưu tiên học đặc trưng của chip bình thường. Khi nhận ảnh mới, mô hình đánh giá mức độ khác biệt giữa ảnh đầu vào và đặc trưng bình thường đã học.

Đầu vào:

- Một ảnh chip cần kiểm tra.
- Mô hình đã được huấn luyện trên tập ảnh Normal.

Đầu ra:

- `true`: Ảnh được đánh giá là Normal.
- `false`: Ảnh được đánh giá là Anomaly.

### 2.2. Logic quyết định

Mô hình Anomaly Detection thường trả về anomaly score thay vì Boolean trực tiếp. Hệ thống sử dụng threshold để chuyển score thành kết quả cuối cùng:

\[
result =
\begin{cases}
true, & score \leq \tau \\
false, & score > \tau
\end{cases}
\]

Trong đó:

- `score`: Điểm bất thường do mô hình tính toán.
- `τ`: Ngưỡng quyết định được lựa chọn trên tập validation.

Threshold không nên được mặc định là `0.5`; giá trị phù hợp phụ thuộc vào mô hình, dữ liệu và yêu cầu cân bằng giữa việc bỏ sót lỗi và cảnh báo nhầm.

### 2.3. Phạm vi phát hiện

Tùy theo đặc điểm ảnh và dữ liệu thực tế, các dạng bất thường có thể bao gồm:

- Thiếu linh kiện.
- Sai loại linh kiện.
- Linh kiện sai vị trí hoặc bị lệch.
- Biến dạng hoặc hư hỏng nhìn thấy được trên bề mặt.
- Những khác biệt bất thường khác so với ảnh chip bình thường.

Ở phiên bản đầu, hệ thống chỉ cần phân loại toàn ảnh thành Normal hoặc Anomaly. Việc xác định chính xác loại lỗi và vị trí lỗi là phần mở rộng, không phải yêu cầu bắt buộc của MVP.

---

## 3. Phương pháp tiếp cận

### 3.1. Normal-only Anomaly Detection

Phương pháp chính được đề xuất là **PatchCore**, triển khai thông qua thư viện Anomalib.

PatchCore sử dụng feature extractor đã được pretrained để trích xuất đặc trưng từ các vùng nhỏ (*patches*) của ảnh bình thường. Những đặc trưng đại diện được lưu trong memory bank. Khi inference, đặc trưng của ảnh mới được so sánh với memory bank để ước lượng mức độ bất thường.

Quy trình khái quát:

**Training**

1. Thu thập ảnh chip bình thường.
2. Tiền xử lý ảnh theo cấu hình thống nhất.
3. Trích xuất feature từ backbone.
4. Tạo patch-level features.
5. Xây dựng memory bank từ đặc trưng bình thường.
6. Lưu model và cấu hình phục vụ inference.

**Inference**

1. Nhận ảnh đầu vào.
2. Tiền xử lý ảnh.
3. Trích xuất feature bằng backbone.
4. So sánh feature với memory bank.
5. Tính anomaly score và anomaly map.
6. So sánh anomaly score với threshold.
7. Trả về Boolean result.

> **Giới hạn dữ liệu hiện tại & Quy chuẩn mở rộng Dataset:** Dự án hiện có hai ảnh mẫu (`image/sample1.jpg` và `image/sample2.jpg` — module cảm biến hồng ngoại LM393). Hai ảnh này đủ để kiểm tra luồng xử lý ở mức prototype, nhưng để mô hình đạt độ chính xác thực tế ($\text{AUROC} \ge 0.92$) và chống nhiễu ánh sáng khi đổi phòng demo, cần thu thập bộ dữ liệu theo chuẩn:
> - **Tập Train (`normal/train`):** **30 – 50 ảnh Normal** chụp cố định góc **90° từ trên xuống (Top-down)**, có xê dịch vị trí nhẹ trên mặt phẳng 2D và chia thành **3–4 điều kiện hướng sáng khác nhau** (ánh sáng phòng, đèn chiếu trái, đèn chiếu phải, hơi yếu sáng) kết hợp `ColorJitter` nhẹ lúc train.
> - **Tập Val/Test (`normal/val` & `abnormal/test`):** **10 – 15 ảnh Normal** và **10 – 15 ảnh Anomaly** (gỡ linh kiện SMD, che linh kiện bằng băng dính nhỏ 1–2mm, đặt lệch linh kiện) dùng để tính ngưỡng $\tau$ tối ưu và đánh giá độc lập.

### 3.2. Tiền xử lý ảnh (Bắt buộc trong Pipeline)

Qua ảnh mẫu thực tế, bo mạch chỉ chiếm <10% diện tích khung hình trong khi lỗi linh kiện dán SMD rất nhỏ. Do đó, các bước tiền xử lý **bắt buộc** trước khi đưa vào PatchCore bao gồm:

1. **Đọc và kiểm tra định dạng ảnh** (chuẩn hóa sang không gian màu RGB).
2. **Tách nền thích nghi chống bóng đổ (Adaptive Board Segmentation):** Chuyển sang hệ màu LAB/HSV kết hợp ngưỡng tự động Otsu và phép đóng hình thái học (`MORPH_CLOSE`) để tách riêng phần bo mạch tối màu khỏi nền giấy sáng.
3. **Tự động cắt sát & Xoay thẳng bo mạch (Auto-ROI Crop & Deskew):** Tìm contour lớn nhất của bo mạch, dùng `cv2.minAreaRect` và phép biến đổi Affine để dựng thẳng đứng bo mạch (`Height > Width`) và cắt sát viền (kèm lề đệm ~2%).
4. **Định hướng đầu-cuối (180° Orientation Lock):** Đảm bảo chiều của bo mạch luôn đồng nhất (cụm 2 đèn LED ở trên, 3 chân cắm kim loại ở dưới), tự động xoay 180° nếu người dùng đặt ngược đầu.
5. **Chuẩn hóa độ sáng cục bộ (Illumination Normalization):** Áp dụng cân bằng trắng Gray-World và cân bằng histogram cục bộ nhẹ (**CLAHE trên kênh L của hệ màu LAB**, `clipLimit=1.5`) trước khi resize về `(256, 256)`.

### 3.3. Vai trò của các công nghệ & Môi trường triển khai

- **PatchCore (`wide_resnet50_2` backbone):** Mô hình Anomaly Detection chính, giữ nguyên backbone chuẩn mạnh mẽ để phát hiện chi tiết linh kiện SMD nhỏ.
- **Anomalib & PyTorch:** Framework hỗ trợ huấn luyện trên **Google Colab (GPU)** và xuất trọng số (`.ckpt`) về chạy trực tiếp trên **Laptop cá nhân**.
- **OpenCV:** Tách nền, cắt & xoay thẳng bo mạch, chuẩn hóa sáng CLAHE và thu nhận frame từ camera.
- **Streamlit:** Giao diện demo tích hợp cả hai giai đoạn kèm thanh trượt tinh chỉnh nóng ngưỡng $\tau$ (Threshold Slider).

YOLO chưa phải thành phần bắt buộc trong MVP. YOLO có thể được bổ sung ở giai đoạn sau nếu cần nhận diện, phân loại và định vị từng linh kiện cụ thể. Việc phát hiện anomaly toàn ảnh và việc nhận diện từng linh kiện là hai nhiệm vụ khác nhau.

---

## 4. Phase 1 – Image-based Inspection

### 4.1. Mục tiêu

Xây dựng ứng dụng cho phép người dùng tải một ảnh chip lên, thực hiện kiểm tra và nhận kết quả `true` hoặc `false`.

### 4.2. Quy trình

```text
User uploads image
        |
        v
Image loading
        |
        v
Image preprocessing
        |
        v
PatchCore inference
        |
        v
Anomaly score calculation
        |
        v
Threshold comparison
        |
        v
Boolean result
   /          \
true          false
Normal        Anomaly
```

### 4.3. Giao diện

Giao diện Phase 1 được xây dựng bằng Streamlit, gồm:

- Khu vực upload ảnh.
- Nút `Inspect`.
- Hiển thị ảnh đầu vào.
- Kết quả `TRUE` hoặc `FALSE`.
- Anomaly score.
- Anomaly heatmap (nếu có).
- Thời gian xử lý ảnh.

### 4.4. Tiêu chí hoàn thành

Phase 1 được xem là hoàn thành khi:

- Có thể tải ảnh hợp lệ lên ứng dụng.
- Ảnh được đưa qua pipeline preprocessing và model inference.
- Hệ thống trả về Boolean result theo threshold.
- Có thể hiển thị anomaly score và heatmap để hỗ trợ quan sát.
- Có quy trình đánh giá trên tập test độc lập.

---

## 5. Phase 2 – Camera-based Inspection

### 5.1. Mục tiêu

Mở rộng hệ thống để nhận ảnh trực tiếp từ camera. Khi chip xuất hiện trong vùng kiểm tra, hệ thống tự động thu nhận ảnh, thực hiện inference và hiển thị kết quả.

Phase 2 **tái sử dụng model và inspection pipeline của Phase 1**. Thành phần được bổ sung chủ yếu là camera acquisition và cơ chế trigger.

### 5.2. Quy trình

```text
Camera starts
      |
      v
Read video frames
      |
      v
Detect chip in inspection ROI
      |
      v
Wait until chip is stable
      |
      v
Capture inspection frame
      |
      v
Shared inspection pipeline
      |
      v
PatchCore inference
      |
      v
Anomaly score + threshold
      |
      v
Boolean result
   /          \
true          false
Normal        Anomaly
```

### 5.3. Thu nhận hình ảnh và trigger

Có thể triển khai theo mức độ tăng dần:

1. **Manual capture:** Người dùng nhấn nút để chụp ảnh. Đây là cách đơn giản nhất để kiểm tra camera integration.
2. **ROI / motion-based trigger:** OpenCV theo dõi một vùng cố định trong frame. Khi chip xuất hiện và ổn định qua một số frame liên tiếp, hệ thống chụp ảnh.
3. **Sensor trigger:** Nếu có thiết bị phù hợp, cảm biến quang điện có thể phát tín hiệu khi chip đi vào vùng kiểm tra.

Đối với đồ án, có thể dùng webcam USB và ROI-based trigger. Nếu không có băng chuyền thật, có thể mô phỏng việc đưa chip vào vùng quan sát bằng cách đặt chip trước camera.

Cần có cơ chế tránh kiểm tra lặp lại cùng một chip ở nhiều frame. Ví dụ: sau khi capture, hệ thống chỉ cho phép trigger lần tiếp theo khi chip hiện tại đã rời khỏi ROI.

### 5.4. Tiêu chí hoàn thành

- Camera có thể kết nối và cung cấp frame.
- Hệ thống hiển thị được luồng hình ảnh.
- Có thể capture ảnh thủ công hoặc tự động theo trigger.
- Ảnh capture được xử lý bởi cùng pipeline của Phase 1.
- Kết quả `true` hoặc `false` được hiển thị sau mỗi lần kiểm tra.
- Hệ thống không liên tục kiểm tra lặp cùng một chip.

---

## 6. Kiến trúc tổng thể

Thiết kế hệ thống theo hướng tách biệt **Image Acquisition** và **Inspection Pipeline**.

```text
                 IMAGE ACQUISITION
                 /                \
        Image Upload             Camera
              |                    |
              v                    v
         Image Loader         Frame Capture
              |                    |
              +---------+----------+
                        |
                        v
                INSPECTION PIPELINE
                        |
                        v
                Image Preprocessing
                        |
                        v
                 PatchCore Model
                        |
                        v
                 Anomaly Score
                        |
                        v
                Threshold Decision
                        |
                        v
                 Boolean Result
                        |
                        v
                  Streamlit UI
```

Thiết kế này giúp chuyển từ Phase 1 sang Phase 2 mà không cần thay đổi model hoặc viết lại logic kiểm tra. Hai nguồn ảnh chỉ cần cung cấp ảnh theo cùng một định dạng cho inspection pipeline.

### 6.1. Giao diện model thống nhất

Có thể xây dựng một lớp model wrapper để đóng gói thao tác load model và inference:

```python
class InspectionModel:
    def load(self, model_path):
        ...

    def predict(self, image):
        ...
```

Một kết quả inference thống nhất có thể có dạng:

```python
{
    "is_normal": True,
    "anomaly_score": 0.12,
    "threshold": 0.50,
    "anomaly_map": None,
    "processing_time_ms": 35
}
```

Các giá trị trong ví dụ chỉ minh họa cấu trúc dữ liệu, không phải kết quả benchmark. `is_normal` là Boolean dùng cho giao diện; anomaly score và threshold được giữ lại để kiểm tra, hiệu chỉnh và đánh giá (hỗ trợ thanh trượt tinh chỉnh nóng $\tau$ trực tiếp trên Sidebar của Streamlit khi demo thực tế).

---

## 7. Tài liệu đặc tả kỹ thuật cho Team Dev

Toàn bộ thiết kế chi tiết về cấu trúc thư mục, thuật toán tiền xử lý OpenCV, quy chuẩn chụp Dataset đa hướng sáng, kịch bản huấn luyện trên Google Colab, hợp đồng dữ liệu (`src/schemas.py`) và phân chia công việc cho từng thành viên được quy định tại:

- **[`docs/developer_spec.md`](./developer_spec.md)**

