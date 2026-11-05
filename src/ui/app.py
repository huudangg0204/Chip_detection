"""Streamlit Web UI for Phase 1 (Image Upload), Phase 2 (Camera Trigger), and Dataset Helper (Dev 3)."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root is on sys.path when launched via `streamlit run src/ui/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition.camera_trigger import CameraTrigger, TriggerState
from src.acquisition.image_loader import load_image_from_bytes, load_image_from_path
from src.inference.inspector import InspectionModel, update_threshold_decision
from src.preprocessing.board_aligner import BoardAligner, preprocess_board_image
from src.schemas import InspectionResult, PreprocessResult

logger = logging.getLogger(__name__)

DATASET_SUBDIRS: dict[str, Path] = {
    "normal_train": Path("data/chip_dataset/normal/train"),
    "normal_val": Path("data/chip_dataset/normal/val"),
    "abnormal_test": Path("data/chip_dataset/abnormal/test"),
}


def draw_board_bbox_overlay(
    image_rgb: np.ndarray,
    bbox_box_points: np.ndarray | None,
    color: tuple[int, int, int] = (34, 197, 94),
    thickness: int = 4,
    pcb_bbox_points: np.ndarray | None = None,
) -> np.ndarray:
    """Draw detected minAreaRect bounding box polygon onto a copy of the raw RGB image."""
    canvas = image_rgb.copy()
    if bbox_box_points is None or bbox_box_points.size == 0:
        return canvas

    h, w = canvas.shape[:2]
    scaled_thickness = max(thickness, round(max(h, w) / 400))

    if pcb_bbox_points is not None and pcb_bbox_points.size > 0:
        pcb_pts = np.asarray(pcb_bbox_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas,
            [pcb_pts],
            isClosed=True,
            color=(59, 130, 246),
            thickness=max(2, scaled_thickness - 1),
        )

    pts = np.asarray(bbox_box_points, dtype=np.int32).reshape((-1, 1, 2))
    # Adapt line thickness to image resolution so high-res photos have visible borders
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=scaled_thickness)
    return canvas


def draw_camera_roi_overlay(
    frame_rgb: np.ndarray,
    trigger: CameraTrigger,
    state_info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw ROI guide rectangle and state status onto a live camera RGB frame."""
    canvas = frame_rgb.copy()
    x1, y1, x2, y2 = trigger.get_roi_pixel_coords(canvas.shape)

    state_str = str((state_info or {}).get("state", trigger.state.value))
    if state_str == TriggerState.STABILIZING.value:
        color = (234, 179, 8)  # Amber: stabilizing
    elif state_str in (TriggerState.TRIGGER_CAPTURE.value, TriggerState.COOLDOWN_WAIT_EXIT.value):
        color = (34, 197, 94)  # Green: captured / cooldown
    else:
        color = (59, 130, 246)  # Blue: idle waiting for board

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

    # Corner markers for clear visual alignment
    corner_len = max(14, min(x2 - x1, y2 - y1) // 10)
    for cx, cy, dx, dy in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(canvas, (cx, cy), (cx + dx * corner_len, cy), color, 5)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * corner_len), color, 5)

    label = format_trigger_status_text(trigger.state, state_info)
    cv2.putText(
        canvas,
        label,
        (max(10, x1), max(28, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def format_trigger_status_text(
    state: TriggerState | str,
    info: dict[str, Any] | None = None,
) -> str:
    """Return human-readable Vietnamese status label for CameraTrigger state machine."""
    state_val = state.value if isinstance(state, TriggerState) else str(state)
    info_dict = info or {}
    stable_cnt = int(info_dict.get("stable_counter", 0))
    req_cnt = int(info_dict.get("required_stable", 10))

    if state_val == TriggerState.IDLE_EMPTY.value:
        return "Đang chờ đặt mạch vào vùng ROI..."
    if state_val == TriggerState.STABILIZING.value:
        return f"Đang lấy nét ổn định ({stable_cnt}/{req_cnt})..."
    if state_val == TriggerState.TRIGGER_CAPTURE.value:
        return "Đã chụp & Kiểm tra!"
    if state_val == TriggerState.COOLDOWN_WAIT_EXIT.value:
        return "Đã chụp & Kiểm tra! Nhấc mạch ra để tiếp tục"
    return state_val


def format_inspection_badge_html(result: InspectionResult) -> str:
    """Build accessible HTML badge displaying Boolean TRUE/FALSE inspection outcome."""
    board_lbl = "OK" if result.board_ok is True else ("FAIL" if result.board_ok is False else "N/A")
    pins_lbl = "OK" if result.pins_ok is True else ("FAIL" if result.pins_ok is False else "N/A")
    if result.status == "NO_BOARD_DETECTED":
        bg_color = "#fef3c7"
        border_color = "#d97706"
        text_color = "#92400e"
        headline = "NO BOARD DETECTED — KHÔNG TÌM THẤY BO MẠCH"
        subtext = result.message or "Vui lòng đặt bo mạch rõ nét trên nền sáng trong khung hình."
    elif result.status != "OK" or result.is_normal is None:
        bg_color = "#fee2e2"
        border_color = "#dc2626"
        text_color = "#991b1b"
        headline = "ERROR — LỖI XỬ LÝ"
        subtext = result.message or "Đã xảy ra lỗi trong quá trình kiểm tra."
    elif result.is_normal:
        bg_color = "#dcfce7"
        border_color = "#16a34a"
        text_color = "#166534"
        headline = "TRUE — NORMAL (BÌNH THƯỜNG)"
        subtext = (
            f"Boolean: true | Board: {board_lbl} | 3 Chân: {pins_lbl} "
            f"| Score: {result.anomaly_score:.4f} ≤ Threshold τ: {result.threshold:.4f} "
            f"| Thời gian: {result.processing_time_ms:.1f} ms"
        )
    else:
        bg_color = "#fee2e2"
        border_color = "#dc2626"
        text_color = "#991b1b"
        headline = "FALSE — ANOMALY (PHÁT HIỆN LỖI)"
        subtext = (
            f"Boolean: false | Board: {board_lbl} | 3 Chân: {pins_lbl} "
            f"| Score: {result.anomaly_score:.4f} vs Threshold τ: {result.threshold:.4f} "
            f"| Thời gian: {result.processing_time_ms:.1f} ms — {result.message}"
        )

    return (
        f'<div role="status" aria-live="polite" style="'
        f"background-color: {bg_color}; "
        f"border: 2px solid {border_color}; "
        f"color: {text_color}; "
        f"padding: 1rem 1.25rem; "
        f"border-radius: 0.5rem; "
        f'margin-bottom: 1rem;">'
        f'<div style="font-size: 1.35rem; font-weight: 700; letter-spacing: 0.01em;">{headline}</div>'
        f'<div style="font-size: 0.95rem; margin-top: 0.25rem; font-weight: 500;">{subtext}</div>'
        f"</div>"
    )


def recalibrate_history_entries(
    history: list[dict[str, Any]],
    new_threshold: float,
) -> list[dict[str, Any]]:
    """Update inspection history records in-place when user adjusts live threshold slider tau."""
    updated: list[dict[str, Any]] = []
    tau = round(float(new_threshold), 4)
    for item in history:
        entry = dict(item)
        score = entry.get("anomaly_score")
        if isinstance(score, (int, float)):
            is_norm = bool(float(score) <= tau)
            entry["threshold"] = tau
            entry["boolean_result"] = is_norm
            entry["result_label"] = "TRUE (Normal)" if is_norm else "FALSE (Anomaly)"
        updated.append(entry)
    return updated


def get_dataset_counts(dataset_root: Path = Path("data/chip_dataset")) -> dict[str, int]:
    """Count existing images in normal/train, normal/val, and abnormal/test directories."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}
    for key, rel_dir in [
        ("normal_train", Path("normal/train")),
        ("normal_val", Path("normal/val")),
        ("abnormal_test", Path("abnormal/test")),
    ]:
        folder = dataset_root / rel_dir
        if folder.exists():
            counts[key] = sum(
                1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
            )
        else:
            counts[key] = 0
    return counts


def save_dataset_image(
    image_rgb: np.ndarray,
    subset_key: str,
    dataset_root: Path = Path("data/chip_dataset"),
    prefix: str | None = None,
) -> Path:
    """Save an RGB image into the target Anomalib dataset subfolder and return saved file path."""
    rel_map = {
        "normal_train": Path("normal/train"),
        "normal_val": Path("normal/val"),
        "abnormal_test": Path("abnormal/test"),
    }
    if subset_key not in rel_map:
        raise ValueError(f"Invalid dataset subset_key: {subset_key}")

    target_dir = dataset_root / rel_map[subset_key]
    target_dir.mkdir(parents=True, exist_ok=True)

    tag = prefix or subset_key
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = target_dir / f"{tag}_{timestamp}.jpg"

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out_path), bgr):
        raise OSError(f"Không thể lưu ảnh vào {out_path}")
    return out_path


def _detect_runtime_device(inspector: InspectionModel) -> tuple[str, str]:
    """Return (backend_name, device_name) for sidebar status display."""
    extractor = getattr(inspector, "_feature_extractor", None)
    if extractor is not None:
        backend = str(getattr(extractor, "backend_type", "fallback_cv"))
        device_obj = getattr(extractor, "_device", None)
        device_str = str(device_obj).upper() if device_obj is not None else "CPU"
        return backend, device_str
    return "fallback_cv", "CPU"


def _first_existing_path(*candidates: str) -> Path | None:
    """Return the first candidate file path that exists on disk (prefers bundled samples/)."""
    for rel in candidates:
        p = Path(rel)
        if p.is_file():
            return p
    return None


def _render_inspection_details(
    st: Any,
    raw_rgb: np.ndarray,
    prep_result: PreprocessResult,
    insp_result: InspectionResult,
) -> None:
    """Render the 3-column visual comparison, prominent Boolean badge, and 3-pin inspection section."""
    st.markdown(format_inspection_badge_html(insp_result), unsafe_allow_html=True)

    warnings_list = list(getattr(insp_result, "warnings", None) or getattr(prep_result, "warnings", None) or [])
    if warnings_list:
        for warn_text in warnings_list:
            st.warning(f"⚠️ Cảnh báo tiền xử lý: {warn_text}")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    bool_str = (
        "N/A"
        if insp_result.is_normal is None
        else ("TRUE (Normal)" if insp_result.is_normal else "FALSE (Anomaly)")
    )
    score_str = (
        "N/A" if insp_result.anomaly_score is None else f"{insp_result.anomaly_score:.4f}"
    )
    m_col1.metric("Kết quả Boolean", bool_str)
    m_col2.metric("Anomaly Score", score_str)
    m_col3.metric("Threshold (τ)", f"{insp_result.threshold:.4f}")
    m_col4.metric("Processing Time", f"{insp_result.processing_time_ms:.1f} ms")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Ảnh gốc (Raw Image)")
        boxed_rgb = draw_board_bbox_overlay(
            raw_rgb,
            prep_result.bbox_box_points,
            pcb_bbox_points=getattr(prep_result, "pcb_bbox_points", None),
        )
        st.image(
            boxed_rgb,
            caption="Khung xanh lá: Full ROI (gồm 3 chân) | Xanh dương: Thân PCB",
            use_container_width=True,
        )

    with col2:
        st.subheader("2. Sau Preprocessing (Aligned ROI)")
        if prep_result.normalized_rgb is not None:
            margin_val = float(getattr(prep_result, "pin_end_margin", 0.0) or 0.0)
            st.image(
                prep_result.normalized_rgb,
                caption=(
                    f"Đã xoay thẳng ({prep_result.rotation_angle:.1f}°), xóa nền/bóng đổ "
                    f"& CLAHE (256×256) | Pin Margin: {margin_val * 100:.1f}%"
                ),
                use_container_width=True,
            )
        else:
            st.warning(prep_result.error_message or "Không tách được bo mạch.")

    with col3:
        st.subheader("3. Bản đồ nhiệt (Anomaly Heatmap)")
        if insp_result.heatmap_overlay_rgb is not None:
            st.image(
                insp_result.heatmap_overlay_rgb,
                caption="Bản đồ nhiệt PatchCore (Đỏ/Vàng: Vùng nghi ngờ lỗi trên mạch)",
                use_container_width=True,
            )
        else:
            st.info("Chưa có bản đồ nhiệt (không tìm thấy bo mạch hoặc chưa chạy inference).")

    pin_res = getattr(insp_result, "pin_result", None)
    if pin_res is not None:
        st.divider()
        st.subheader("4. Kiểm tra riêng 3 chân (Pin Inspector)")
        p_col1, p_col2 = st.columns([1, 1.8])
        with p_col1:
            if pin_res.overlay_rgb is not None:
                st.image(
                    pin_res.overlay_rgb,
                    caption=f"Vùng 3 chân (256×256) — Phát hiện: {pin_res.pin_count}/3 chân",
                    use_container_width=True,
                )
        with p_col2:
            if pin_res.pins_ok:
                st.success(f"✅ {pin_res.message}")
            else:
                st.error(f"❌ {pin_res.message}")
            if pin_res.per_pin:
                table_rows = [
                    {
                        "Chân": p.get("pin_name", ""),
                        "Trạng thái": "OK" if p.get("status") == "OK" else "NG",
                        "Chiều dài (px)": p.get("length_px", 0.0),
                        "Độ rộng (px)": p.get("width_px", 0.0),
                        "Góc lệch (°)": p.get("angle_deg", 0.0),
                        "Khoảng cách (px)": p.get("spacing_px", "-"),
                        "Chi tiết": p.get("reason_str", "OK"),
                    }
                    for p in pin_res.per_pin
                ]
                st.dataframe(table_rows, use_container_width=True, hide_index=True)


def main() -> None:
    """Entry point for `streamlit run src/ui/app.py`."""
    import streamlit as st

    st.set_page_config(
        page_title="Chip Anomaly Detection System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Chip Anomaly Detection System (PatchCore)")
    st.caption(
        "Hệ thống Kiểm tra Bất thường Bo mạch Cảm biến LM393 — Phase 1 (Upload Ảnh) & Phase 2 (Camera Auto-Trigger)"
    )

    # Initialize cached model in session_state
    if "inspector" not in st.session_state:
        inspector = InspectionModel(
            weights_path=Path("weights/patchcore_best.ckpt"),
            metadata_path=Path("weights/metadata.json"),
            config_path=Path("configs/patchcore_chip.yaml"),
        )
        if inspector.weights_path.exists():
            try:
                inspector.load()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load weights on startup: %s", exc)
        st.session_state["inspector"] = inspector

    if "camera_trigger" not in st.session_state:
        st.session_state["camera_trigger"] = CameraTrigger.from_config("configs/patchcore_chip.yaml")

    if "phase2_history" not in st.session_state:
        st.session_state["phase2_history"] = []

    inspector: InspectionModel = st.session_state["inspector"]
    trigger: CameraTrigger = st.session_state["camera_trigger"]

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("Bảng Điều Khiển & Hiệu Chỉnh")

        st.subheader("1. Trạng thái Model (Model Status)")
        weights_ok = inspector.weights_path.exists() and inspector.is_loaded
        backend_name, device_name = _detect_runtime_device(inspector)
        if weights_ok:
            st.success(f"Đã nạp trọng số: `{inspector.weights_path}`")
        else:
            st.error(f"Chưa nạp trọng số: `{inspector.weights_path}`")
        st.markdown(
            f"- **Backbone cấu hình:** `{inspector.backbone_name}`\n"
            f"- **Feature Backend:** `{backend_name}`\n"
            f"- **Thiết bị thực thi:** `{device_name}`"
        )

        st.divider()
        st.subheader("2. Tinh chỉnh nóng Ngưỡng (Threshold τ)")
        live_tau = float(
            st.slider(
                "Ngưỡng quyết định (Threshold τ)",
                min_value=0.0,
                max_value=1.0,
                value=float(round(inspector.default_threshold, 2)),
                step=0.01,
                help="Kéo thanh trượt để cập nhật ngay lập tức kết quả TRUE/FALSE (score <= τ) mà không cần chạy lại model.",
            )
        )

        st.divider()
        st.subheader("3. Cấu hình Tiền xử lý (Preprocessing)")
        apply_clahe_opt = st.checkbox(
            "Bật chuẩn hóa sáng CLAHE & White Balance",
            value=True,
            help="Cân bằng trắng Gray-World kết hợp CLAHE trên kênh L (hệ màu LAB) giúp chống đổi nguồn sáng.",
        )
        clahe_clip_opt = float(
            st.slider(
                "CLAHE Clip Limit",
                min_value=0.5,
                max_value=4.0,
                value=1.5,
                step=0.1,
            )
        )
        min_area_opt = float(
            st.slider(
                "Độ nhạy diện tích mạch tối thiểu (Min Area Ratio)",
                min_value=0.005,
                max_value=0.10,
                value=0.01,
                step=0.005,
                format="%.3f",
            )
        )
        pin_sensitivity_opt = float(
            st.slider(
                "Độ nhạy kiểm tra 3 chân (Pin Sensitivity)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.05,
                help="Hệ số độ nhạy khi kiểm tra chiều dài, độ rộng, góc lệch và khoảng cách 3 chân cắm (cao hơn = khắt khe hơn).",
            )
        )

        # Apply live preprocessing parameters to aligner
        if isinstance(inspector.aligner, BoardAligner):
            inspector.aligner.apply_clahe = apply_clahe_opt
            inspector.aligner.apply_clahe_flag = apply_clahe_opt
            inspector.aligner.clahe_clip_limit = clahe_clip_opt
            inspector.aligner.min_area_ratio = min_area_opt
        trigger.min_area_ratio = min_area_opt

    # ==================== 3 TABS CHỨC NĂNG ====================
    tab1, tab2, tab3 = st.tabs(
        [
            "Phase 1: Kiểm tra qua Ảnh tải lên",
            "Phase 2: Kiểm tra qua Camera (Auto / Manual)",
            "Công cụ Chụp & Quản lý Dataset",
        ]
    )

    # -------------------- TAB 1: PHASE 1 (IMAGE UPLOAD) --------------------
    with tab1:
        st.subheader("Phase 1 – Image-based Inspection")
        src_col1, src_col2 = st.columns([2, 1])

        with src_col1:
            uploaded_file = st.file_uploader(
                "Tải ảnh bo mạch lên (.jpg, .jpeg, .png)",
                type=["jpg", "jpeg", "png"],
                key="phase1_uploader",
            )
        with src_col2:
            quick_samples: list[tuple[str, Path | None]] = [
                ("Tự tải ảnh lên", None),
            ]
            for label, cand_paths in [
                (
                    "Ảnh thực tế Normal (normal_val_001.jpg)",
                    ("samples/normal_val_001.jpg", "data/chip_dataset/normal/val/normal_val_001.jpg"),
                ),
                (
                    "Ảnh thực tế Anomaly — Che biến trở 103 (anomaly_test_005.jpg)",
                    ("samples/anomaly_test_005.jpg", "data/chip_dataset/abnormal/test/anomaly_test_005.jpg"),
                ),
                (
                    "Ảnh thực tế Anomaly — Che IC LM393 (anomaly_test_007.jpg)",
                    ("samples/anomaly_test_007.jpg", "data/chip_dataset/abnormal/test/anomaly_test_007.jpg"),
                ),
                (
                    "Ảnh thực tế Anomaly — Che LED (anomaly_test_001.jpg)",
                    ("samples/anomaly_test_001.jpg", "data/chip_dataset/abnormal/test/anomaly_test_001.jpg"),
                ),
                (
                    "Ảnh thực tế Anomaly — Dị vật / Che chân cắm (anomaly_test_003.jpg)",
                    ("samples/anomaly_test_003.jpg", "data/chip_dataset/abnormal/test/anomaly_test_003.jpg"),
                ),
                (
                    "Ảnh thực tế Anomaly — Cong chân cắm (anomaly_test_011.jpg)",
                    ("samples/anomaly_test_011.jpg", "data/chip_dataset/abnormal/test/anomaly_test_011.jpg"),
                ),
                ("Mẫu gốc Normal (image/sample2.jpg)", ("image/sample2.jpg",)),
                ("Mẫu gốc Anomaly (image/sample1.jpg)", ("image/sample1.jpg",)),
            ]:
                resolved = _first_existing_path(*cand_paths)
                if resolved is not None:
                    quick_samples.append((label, resolved))

            sample_choice = st.selectbox(
                "Hoặc chọn nhanh ảnh trong bộ Dataset / Mẫu:",
                options=quick_samples,
                index=1 if (uploaded_file is None and len(quick_samples) > 1) else 0,
                format_func=lambda x: x[0],
            )
            inspect_btn = st.button("Inspect (Phân tích ảnh)", type="primary", use_container_width=True)

        selected_rgb: np.ndarray | None = None
        source_tag = ""

        if uploaded_file is not None and sample_choice[1] is None:
            try:
                selected_rgb = load_image_from_bytes(uploaded_file.getvalue())
                source_tag = f"upload:{uploaded_file.name}:{len(uploaded_file.getvalue())}"
            except ValueError as exc:
                st.error(f"Lỗi đọc file ảnh tải lên: {exc}")
        elif sample_choice[1] is not None and sample_choice[1].exists():
            selected_rgb = load_image_from_path(sample_choice[1])
            source_tag = str(sample_choice[1])

        # Run model inference when source changes, preprocessing settings change, or Inspect button clicked
        prep_sig = f"{source_tag}|{apply_clahe_opt}|{clahe_clip_opt:.2f}|{min_area_opt:.3f}|{pin_sensitivity_opt:.2f}"
        if selected_rgb is not None and (
            inspect_btn
            or st.session_state.get("phase1_last_sig") != prep_sig
            or "phase1_result" not in st.session_state
        ):
            prep_res = preprocess_board_image(selected_rgb, aligner=inspector.aligner)
            insp_res = inspector.predict(
                selected_rgb,
                threshold=live_tau,
                preprocess_result=prep_res,
                pin_sensitivity=pin_sensitivity_opt,
            )
            st.session_state["phase1_raw_rgb"] = selected_rgb
            st.session_state["phase1_prep_result"] = prep_res
            st.session_state["phase1_result"] = insp_res
            st.session_state["phase1_last_sig"] = prep_sig

        if "phase1_result" in st.session_state and "phase1_raw_rgb" in st.session_state:
            # Instant threshold recalibration without re-running model inference!
            live_result = update_threshold_decision(st.session_state["phase1_result"], live_tau)
            st.session_state["phase1_result"] = live_result
            _render_inspection_details(
                st,
                st.session_state["phase1_raw_rgb"],
                st.session_state["phase1_prep_result"],
                live_result,
            )
        else:
            st.info("Hãy tải một ảnh bo mạch lên hoặc chọn ảnh mẫu để bắt đầu kiểm tra.")

    # -------------------- TAB 2: PHASE 2 (CAMERA INSPECTION) --------------------
    with tab2:
        st.subheader("Phase 2 – Camera-based Inspection")
        cam_ctrl1, cam_ctrl2, cam_ctrl3 = st.columns([1, 1, 1])

        with cam_ctrl1:
            camera_source_mode = st.selectbox(
                "Nguồn thu nhận Camera:",
                options=[
                    "Webcam Trình duyệt / Chụp nhanh (st.camera_input)",
                    "USB Webcam OpenCV (Camera Index 0, 1...)",
                    "Mô phỏng luồng Camera từ Ảnh mẫu (Demo Không cần Webcam)",
                ],
            )
        with cam_ctrl2:
            trigger_mode = st.radio(
                "Chế độ kích hoạt (Trigger Mode):",
                options=[
                    "Auto-Trigger (Tự động chụp khi mạch đứng yên N=10 frames)",
                    "Manual Capture (Nút bấm chụp thủ công — An toàn 100%)",
                ],
            )
        with cam_ctrl3:
            cam_index = int(st.number_input("Chỉ số USB Camera (Index):", min_value=0, max_value=5, value=0, step=1))
            if st.button("Inspect Again / Reset Trigger", use_container_width=True):
                trigger.reset()
                st.toast("Đã reset bộ kích hoạt Camera về trạng thái IDLE_EMPTY.")

        live_col, result_col = st.columns([1.15, 1.35])

        with live_col:
            st.markdown("#### Khung hình Camera & Vùng định vị ROI")
            captured_frame_to_inspect: np.ndarray | None = None
            capture_mode_label = "Auto-Trigger" if "Auto-Trigger" in trigger_mode else "Manual"

            if "st.camera_input" in camera_source_mode:
                cam_photo = st.camera_input("Đặt bo mạch vào giữa khung hình và nhấn chụp")
                if cam_photo is not None:
                    try:
                        frame_rgb = load_image_from_bytes(cam_photo.getvalue())
                        _, _, _, info = trigger.process_frame(frame_rgb)
                        overlay_rgb = draw_camera_roi_overlay(frame_rgb, trigger, info)
                        st.image(overlay_rgb, caption="Khung định vị ROI trên ảnh Camera", use_container_width=True)
                        captured_frame_to_inspect = trigger.force_manual_trigger(frame_rgb)
                    except ValueError as exc:
                        st.error(f"Không thể đọc frame từ camera: {exc}")

            elif "USB Webcam OpenCV" in camera_source_mode:
                btn_c1, btn_c2 = st.columns(2)
                start_stream = btn_c1.button("Quét 15 Frames từ USB Camera", type="primary", use_container_width=True)
                manual_snap = btn_c2.button("Manual Capture (Chụp ngay)", use_container_width=True)

                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                if start_stream or manual_snap:
                    cap = cv2.VideoCapture(cam_index)
                    if not cap.isOpened():
                        st.error(
                            f"Không thể kết nối USB Camera (Index={cam_index}). "
                            "Hãy chuyển sang chế độ Webcam Trình duyệt hoặc Mô phỏng!"
                        )
                    else:
                        try:
                            max_steps = 1 if manual_snap else 20
                            for _ in range(max_steps):
                                ok, frame_bgr = cap.read()
                                if not ok or frame_bgr is None:
                                    break
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                if manual_snap or "Manual" in trigger_mode:
                                    captured_frame_to_inspect = trigger.force_manual_trigger(frame_rgb)
                                    overlay_rgb = draw_camera_roi_overlay(frame_rgb, trigger, {"state": trigger.state.value})
                                    frame_placeholder.image(overlay_rgb, use_container_width=True)
                                    status_placeholder.success(format_trigger_status_text(trigger.state))
                                    break

                                state, should_trig, cap_frame, info = trigger.process_frame(frame_rgb)
                                overlay_rgb = draw_camera_roi_overlay(frame_rgb, trigger, info)
                                frame_placeholder.image(overlay_rgb, use_container_width=True)
                                status_placeholder.info(format_trigger_status_text(state, info))
                                if should_trig and cap_frame is not None:
                                    captured_frame_to_inspect = cap_frame
                                    break
                                time.sleep(0.03)
                        finally:
                            cap.release()

            else:
                sim_options: list[tuple[str, Path | None]] = []
                for label, cand_paths in [
                    (
                        "Bo mạch Normal thực tế (normal_val_001.jpg)",
                        ("samples/normal_val_001.jpg", "data/chip_dataset/normal/val/normal_val_001.jpg"),
                    ),
                    (
                        "Bo mạch Anomaly — Che biến trở 103 (anomaly_test_005.jpg)",
                        ("samples/anomaly_test_005.jpg", "data/chip_dataset/abnormal/test/anomaly_test_005.jpg"),
                    ),
                    (
                        "Bo mạch Anomaly — Che IC LM393 (anomaly_test_007.jpg)",
                        ("samples/anomaly_test_007.jpg", "data/chip_dataset/abnormal/test/anomaly_test_007.jpg"),
                    ),
                    (
                        "Bo mạch Anomaly — Che LED (anomaly_test_001.jpg)",
                        ("samples/anomaly_test_001.jpg", "data/chip_dataset/abnormal/test/anomaly_test_001.jpg"),
                    ),
                    (
                        "Bo mạch Anomaly — Dị vật / Che chân cắm (anomaly_test_003.jpg)",
                        ("samples/anomaly_test_003.jpg", "data/chip_dataset/abnormal/test/anomaly_test_003.jpg"),
                    ),
                    (
                        "Bo mạch Anomaly — Cong chân cắm (anomaly_test_011.jpg)",
                        ("samples/anomaly_test_011.jpg", "data/chip_dataset/abnormal/test/anomaly_test_011.jpg"),
                    ),
                    ("Bo mạch Normal mẫu (image/sample2.jpg)", ("image/sample2.jpg",)),
                    ("Bo mạch Anomaly mẫu (image/sample1.jpg)", ("image/sample1.jpg",)),
                ]:
                    resolved = _first_existing_path(*cand_paths)
                    if resolved is not None:
                        sim_options.append((label, resolved))
                sim_options.append(("Khung ROI trống (Nền giấy trắng — IDLE_EMPTY)", None))

                sim_choice = st.selectbox(
                    "Chọn mạch đưa vào vùng ROI mô phỏng:",
                    options=sim_options,
                    format_func=lambda x: x[0],
                )
                sim_btn1, sim_btn2 = st.columns(2)
                run_auto_sim = sim_btn1.button(
                    "Chạy Auto-Trigger (10 frames ổn định)",
                    type="primary",
                    use_container_width=True,
                )
                run_manual_sim = sim_btn2.button(
                    "Bấm chụp tay (Manual Capture)",
                    use_container_width=True,
                )

                if sim_choice[1] is not None and sim_choice[1].exists():
                    sim_frame = load_image_from_path(sim_choice[1])
                else:
                    sim_frame = np.full((600, 800, 3), 245, dtype=np.uint8)

                if run_auto_sim:
                    trigger.reset()
                    last_info: dict[str, Any] = {}
                    for _ in range(trigger.stable_frames_required):
                        _, should_trig, cap_f, last_info = trigger.process_frame(sim_frame)
                        if should_trig and cap_f is not None:
                            captured_frame_to_inspect = cap_f
                            capture_mode_label = "Auto-Trigger"
                            break
                    overlay_rgb = draw_camera_roi_overlay(sim_frame, trigger, last_info)
                    st.image(overlay_rgb, caption=format_trigger_status_text(trigger.state, last_info), use_container_width=True)
                elif run_manual_sim:
                    captured_frame_to_inspect = trigger.force_manual_trigger(sim_frame)
                    capture_mode_label = "Manual Capture"
                    overlay_rgb = draw_camera_roi_overlay(sim_frame, trigger, {"state": trigger.state.value})
                    st.image(overlay_rgb, caption=format_trigger_status_text(trigger.state), use_container_width=True)
                else:
                    overlay_rgb = draw_camera_roi_overlay(sim_frame, trigger, {"state": trigger.state.value})
                    st.image(overlay_rgb, caption=format_trigger_status_text(trigger.state), use_container_width=True)

            st.caption(f"**Trạng thái State Machine hiện tại:** `{format_trigger_status_text(trigger.state)}`")

        if captured_frame_to_inspect is not None:
            prep_cam = preprocess_board_image(captured_frame_to_inspect, aligner=inspector.aligner)
            insp_cam = inspector.predict(
                captured_frame_to_inspect,
                threshold=live_tau,
                preprocess_result=prep_cam,
                pin_sensitivity=pin_sensitivity_opt,
            )
            st.session_state["phase2_last_raw"] = captured_frame_to_inspect
            st.session_state["phase2_last_prep"] = prep_cam
            st.session_state["phase2_last_result"] = insp_cam

            if insp_cam.status == "OK" and insp_cam.anomaly_score is not None:
                st.session_state["phase2_history"].insert(
                    0,
                    {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "mode": capture_mode_label,
                        "anomaly_score": round(float(insp_cam.anomaly_score), 4),
                        "threshold": round(live_tau, 4),
                        "boolean_result": bool(insp_cam.is_normal),
                        "result_label": "TRUE (Normal)" if insp_cam.is_normal else "FALSE (Anomaly)",
                        "processing_time_ms": insp_cam.processing_time_ms,
                    },
                )

        with result_col:
            st.markdown("#### Kết quả Kiểm tra Gần nhất")
            if "phase2_last_result" in st.session_state and "phase2_last_raw" in st.session_state:
                live_cam_res = update_threshold_decision(st.session_state["phase2_last_result"], live_tau)
                st.session_state["phase2_last_result"] = live_cam_res
                _render_inspection_details(
                    st,
                    st.session_state["phase2_last_raw"],
                    st.session_state["phase2_last_prep"],
                    live_cam_res,
                )
            else:
                st.info("Chưa có frame nào được chụp. Hãy đặt mạch vào vùng ROI hoặc nhấn Manual Capture.")

            st.markdown("#### Lịch sử Kiểm tra trong Phiên (Session History)")
            if st.session_state["phase2_history"]:
                st.session_state["phase2_history"] = recalibrate_history_entries(
                    st.session_state["phase2_history"], live_tau
                )
                st.dataframe(st.session_state["phase2_history"], use_container_width=True)
                if st.button("Xóa lịch sử phiên"):
                    st.session_state["phase2_history"] = []
                    st.rerun()
            else:
                st.caption("Chưa có bản ghi kiểm tra nào trong phiên hiện tại.")

    # -------------------- TAB 3: DATASET HELPER --------------------
    with tab3:
        st.subheader("Công cụ Hỗ trợ Chụp & Kiểm tra Dataset (Dataset Helper)")
        st.markdown(
            "Dùng để chụp bộ dữ liệu chuẩn (**30–50 ảnh Normal Train**, **10–15 ảnh Normal Val**, **10–15 ảnh Abnormal Test**) "
            "và kiểm tra chắc chắn `BoardAligner` cắt thẳng đứng bo mạch trước khi lưu."
        )

        counts = get_dataset_counts(Path("data/chip_dataset"))
        c1, c2, c3 = st.columns(3)
        c1.metric("normal/train (Mục tiêu: 30–50)", f"{counts['normal_train']} ảnh")
        c2.metric("normal/val (Mục tiêu: 10–15)", f"{counts['normal_val']} ảnh")
        c3.metric("abnormal/test (Mục tiêu: 10–15)", f"{counts['abnormal_test']} ảnh")

        st.divider()
        ds_col1, ds_col2 = st.columns(2)
        with ds_col1:
            ds_upload = st.file_uploader(
                "Tải ảnh hoặc chụp ảnh để kiểm tra căn chỉnh trước khi lưu vào Dataset:",
                type=["jpg", "jpeg", "png"],
                key="dataset_helper_uploader",
            )
            ds_cam = st.camera_input("Hoặc chụp trực tiếp từ Webcam:", key="dataset_helper_cam")
            target_subset = st.selectbox(
                "Chọn thư mục đích lưu ảnh:",
                options=[
                    ("normal_train", "data/chip_dataset/normal/train (Mạch Normal dùng để Train)"),
                    ("normal_val", "data/chip_dataset/normal/val (Mạch Normal dùng để tìm ngưỡng τ)"),
                    ("abnormal_test", "data/chip_dataset/abnormal/test (Mạch Anomaly dùng để Test)"),
                ],
                format_func=lambda x: x[1],
            )

        candidate_ds_rgb: np.ndarray | None = None
        if ds_cam is not None:
            candidate_ds_rgb = load_image_from_bytes(ds_cam.getvalue())
        elif ds_upload is not None:
            candidate_ds_rgb = load_image_from_bytes(ds_upload.getvalue())

        with ds_col2:
            if candidate_ds_rgb is not None:
                prep_ds = preprocess_board_image(candidate_ds_rgb, aligner=inspector.aligner)
                if prep_ds.success and prep_ds.normalized_rgb is not None:
                    st.success(
                        f"BoardAligner thành công! Góc xoay: {prep_ds.rotation_angle:.1f}° | "
                        f"Kích thước cắt: {prep_ds.cropped_rgb.shape if prep_ds.cropped_rgb is not None else None}"
                    )
                    p_col1, p_col2 = st.columns(2)
                    p_col1.image(
                        draw_board_bbox_overlay(candidate_ds_rgb, prep_ds.bbox_box_points),
                        caption="Ảnh gốc & Khung phát hiện",
                        use_container_width=True,
                    )
                    p_col2.image(
                        prep_ds.normalized_rgb,
                        caption="Ảnh 256×256 sau căn chỉnh & CLAHE",
                        use_container_width=True,
                    )
                    if st.button("Lưu ảnh vào Dataset", type="primary", use_container_width=True):
                        saved_path = save_dataset_image(
                            candidate_ds_rgb,
                            subset_key=target_subset[0],
                            dataset_root=Path("data/chip_dataset"),
                        )
                        st.success(f"Đã lưu ảnh thành công tại: `{saved_path}`")
                else:
                    st.error(
                        f"Không phát hiện được bo mạch hợp lệ ({prep_ds.error_message}). "
                        "Không nên lưu ảnh này vào Dataset!"
                    )
            else:
                st.info("Hãy chụp hoặc tải lên 1 ảnh ở cột bên trái để xem trước kết quả cắt mạch (`BoardAligner`).")


if __name__ == "__main__":
    main()
