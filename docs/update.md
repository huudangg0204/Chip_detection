=# TASK: Sửa 3 vấn đề trong pipeline kiểm tra bo mạch LM393 (crop thiếu chân, bóng đổ, kiểm tra riêng 3 chân)

## 0. Bối cảnh
Pipeline hiện tại: BoardAligner (tách nền, xoay thẳng, CLAHE) -> PatchCore (anomalib) -> InspectionResult. Tài liệu gốc: docs/developer_spec.md. Lỗi xoay 180° đã được sửa.

Quan sát từ lần chạy thực tế trên mạch Normal:
- ROI sau tiền xử lý cắt mất phần lớn 3 chân kim loại ở đáy.
- ROI chứa vùng bóng đổ bên phải bo mạch, heatmap đỏ đúng vùng bóng. Kết quả sai: FALSE, Score 1.0000 > τ 0.25.
- Khi mạch bị mất chân hoặc cong chân, hệ thống vẫn báo Normal vì 3 chân gần như không đóng góp vào điểm toàn ảnh.

Ràng buộc (theo spec, mục 8):
- Train và Inference dùng chung đúng một hàm tiền xử lý. Màu luôn là RGB ở giao diện hàm.
- Chỉ THÊM field vào src/schemas.py, không đổi tên field có sẵn. Mọi thay đổi schema hoặc kích thước ảnh đầu vào phải được cả nhóm duyệt trước (Ask First).
- Không dùng flip/random crop khi train. Không đưa ảnh lỗi vào normal/train. Không commit file weights.
- Mọi thay đổi crop hoặc kích thước ảnh đồng nghĩa với việc phải train lại PatchCore. Weights cũ không dùng được nữa.

---

## 1. Vấn đề 1: Crop thiếu 3 chân

Giả thuyết nguyên nhân (dev cần xác nhận bằng ảnh debug):
- Otsu trên kênh L chỉ lấy phần PCB đen. Chân kim loại sáng bị xếp vào nền, nên minAreaRect chỉ bao PCB.
- Margin 2% không đủ. Chân dài khoảng 14-16% chiều dài PCB (dev đo lại trên ảnh thật).

Yêu cầu:
1. Tách hai khái niệm: pcb_rect (mask PCB đen) và full_rect (PCB + chân).
2. Chỉ mở rộng sau khi đã biết chiều đúng (chân ở dưới). Kéo dài cạnh phía chân thêm pin_extension_ratio (mặc định 0.20 x chiều dài PCB, đưa vào config). Đầu cảm biến và hai cạnh bên dùng margin 2-3%. Không mở rộng trước bước định hướng, vì sẽ cắt sai khi mạch bị ngược.
3. Nếu phần mở rộng vượt ra ngoài khung ảnh gốc: pad bằng giá trị nền đã ước lượng và đặt warning "board near frame edge". Không được cắt cụt chân âm thầm.
4. Bỏ cách pad đen hai bên để ép về ảnh vuông 256x256 (bo mạch hiện chỉ chiếm ~40% chiều ngang, lãng phí độ phân giải). Xuất ảnh full_board giữ tỉ lệ, kích thước cố định theo config. Đề xuất ban đầu: cao 640 x rộng 144 (tỉ lệ ~4.4:1), cần nhóm duyệt.
5. Thêm trường full_board_rgb và pin_end_margin vào kết quả tiền xử lý (chỉ thêm field).

Nghiệm thu:
- Chạy trên >= 30 ảnh chụp bằng camera inference, ở 3 khoảng cách (gần/vừa/xa): 100% ảnh thấy đủ 3 chân và có nền giấy bên dưới đầu chân.
- Lưu ảnh debug (pcb_rect, full_rect, mask) cho toàn bộ 30 ảnh để review bằng mắt.
- Thêm unit test: đầu mút chân cách mép dưới crop >= 3% chiều cao.

---

## 2. Vấn đề 2: Bóng đổ

Có hai tác động riêng biệt. Phải xử lý cả hai:
(i) Tách nền gộp bóng vào mask bo mạch, nên rect và crop bị rộng sai.
(ii) PatchCore coi bóng là vùng lạ nên đẩy score lên.

Các phương án (theo thứ tự ưu tiên):

A. Chụp/phần cứng (làm ngay, rẻ nhất, hiệu quả nhất)
- Ánh sáng khuếch tán, đối xứng từ trên xuống (hai đèn hai bên, đèn vòng, hoặc đèn bounce lên giấy trắng/trần). Tránh đèn đơn chiếu xiên.
- Cố định vị trí đèn. Khóa exposure, white balance và focus của camera.
- Dùng nền nhám, mờ, sát mặt mạch. Tùy chọn thử thêm: nền màu xanh lá mờ thay giấy trắng để tách bằng hue, vì bóng giữ nguyên hue còn bo mạch thì không.

B. Tách nền chống bóng (phần mềm, BẮT BUỘC)
- Ước lượng độ sáng nền bằng morphological closing hoặc median kernel lớn trên kênh L. Chia ảnh cho nền ước lượng để lấy tỉ lệ sáng (flat-field).
- Bóng trên giấy có tỉ lệ sáng trung bình (~0.45-0.85) và chroma gần như nền. PCB đen thì tối hơn nhiều. Dùng multi-Otsu 3 lớp (nền / bóng / PCB) hoặc 2 ngưỡng, chỉ giữ lớp tối nhất làm mask PCB.
- Làm sạch: giữ component lớn nhất, open/close, fill holes (lỗ tròn trắng, biến trở trắng-xanh). Tính minAreaRect từ mask sạch.
- Sanity check: tỉ lệ khung hình PCB ~4.2 +/- 0.4. Nếu lệch nhiều thì cảnh báo, không đưa vào model.

C. Mask nền sau khi crop
- Tạo board_mask trong tọa độ crop (PCB + chân). Mọi pixel ngoài mask thay bằng một màu nền hằng số trước khi đưa vào model. Làm đồng nhất cho cả train lẫn inference.

D. Mask khi tính điểm (lớp bảo hiểm)
- Nhân anomaly_map với board_mask (erode nhẹ) trước khi lấy max. Bóng lọt crop cũng không làm tăng score.

E. Không khuyến nghị làm chính: đưa ảnh có bóng vào tập train. Cách này làm giảm độ nhạy với lỗi nhỏ. Chỉ cân nhắc nếu A-D không đủ.

F. Nâng cao, tùy chọn: ảnh bất biến chiếu sáng (log-chromaticity). Phức tạp, chỉ thử nếu B không ổn.

Nghiệm thu:
- Chụp >= 20 ảnh Normal cố ý có bóng (đèn trái, đèn phải, bóng đậm, bóng nhạt).
- Mask không chứa bóng (kiểm tra bằng mắt trên toàn bộ, IoU với mask vẽ tay >= 0.95 trên 10 ảnh).
- Heatmap không đỏ ở ngoài bo mạch.
- Chênh lệch score Normal giữa có bóng và không bóng nhỏ (đề xuất <= 0.05 trên thang raw chuẩn hóa, nhóm chốt).

---

## 3. Vấn đề 3: Kiểm tra riêng 3 chân

Vì sao PatchCore toàn bo mạch bỏ qua chân:
- Chân chiếm vài % diện tích, mà score = max trên toàn ảnh nên bị lấn át bởi nhiễu chỗ khác.
- PatchCore không có thông tin vị trí. Khi mất chân, patch còn lại là nền giấy, vốn đã có trong memory bank nên điểm thấp. Chân cong nhẹ vẫn giống patch mép chân. Vì vậy chỉ tách ROI chân rồi chạy PatchCore vẫn có thể bỏ sót.

Thiết kế: tạo src/inference/pin_inspector.py độc lập.

Đầu vào: full_board_rgb + board_mask. Cắt pin_roi từ vùng header trở xuống hết đầu chân (+ nền). Tọa độ định nghĩa theo tỉ lệ trong config, vì sau căn chỉnh vị trí gần như cố định. Pin_roi gần như vuông, resize lên >= 256x256 nên mỗi chân có nhiều pixel hơn hẳn.

Lớp 1 (chủ lực, xác định, giải thích được):
1. Tách chân trong pin_roi: pixel khác nền cục bộ (cùng kỹ thuật tỉ lệ sáng như phần bóng, để loại bóng của chân), open theo chiều dọc.
2. Lấy connected components dạng thanh dọc. Yêu cầu đếm = 3.
3. Đo từng chân: chiều dài, độ rộng, góc lệch so với trục dọc (fitLine hoặc minAreaRect), tọa độ x tâm, khoảng cách giữa các chân.
4. So với ngưỡng học từ tập Normal (mean +/- k*std hoặc min/max nới 20%), lưu trong weights/pin_metadata.json.
5. Trả về lý do cho từng chân: thiếu / ngắn / cong (kèm góc) / lệch khoảng cách.

Lớp 2 (tùy chọn, bổ trợ): mô hình anomaly riêng cho pin_roi. Ưu tiên PaDiM (có thông tin vị trí) hoặc PatchCore chấm điểm theo 3 slot, hoặc so ảnh với template chuẩn sau căn chỉnh. Quyết định bằng thí nghiệm. Bỏ qua Lớp 2 nếu Lớp 1 đạt tiêu chí.

Quyết định cuối: is_normal = board_ok AND pins_ok. UI luôn hiển thị phần nào gây lỗi.

Schema (chỉ thêm, cần nhóm duyệt): pin_result: Optional[PinCheckResult] gồm pins_ok, pin_count, per_pin (list dict số đo + trạng thái), message, pin_roi_rgb, overlay_rgb.

UI (src/ui/app.py): thêm panel "Kiểm tra chân" hiển thị pin_roi với khung xanh/đỏ từng chân và bảng số đo. Thêm slider độ nhạy cho pin check.



Nghiệm thu (đề xuất, nhóm chốt):
- Normal: pass >= 95% trên tập val.
- Lỗi: phát hiện >= 90% thiếu chân và chân cong rõ rệt.
- Báo cáo bảng kết quả riêng từng loại lỗi.

---

## 4. Thứ tự thực hiện
1. Sửa crop (Vấn đề 1), review ảnh debug.
2. Tách nền chống bóng + mask (Vấn đề 2), review với ảnh có bóng.
3. Pin inspector Lớp 1 + tích hợp schema/UI (Vấn đề 3).
4. Chụp lại dataset bằng camera inference, train lại PatchCore trên ảnh đã mask, tính lại τ.
5. Lớp 2 nếu cần.

Deliverables: code, unit tests (crop giữ đủ chân, ảnh bóng giả lập, pin_inspector trên ảnh tổng hợp), ảnh debug, bảng kết quả, cập nhật config