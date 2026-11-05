"""Unit tests for Dev 3 Streamlit UI helpers and state recalibration (src/ui/app.py)."""

from pathlib import Path

import numpy as np
import pytest

from src.acquisition.camera_trigger import CameraTrigger, TriggerState
from src.schemas import InspectionResult
from src.ui.app import (
    draw_board_bbox_overlay,
    draw_camera_roi_overlay,
    format_inspection_badge_html,
    format_trigger_status_text,
    get_dataset_counts,
    recalibrate_history_entries,
    save_dataset_image,
)


def test_draw_board_bbox_overlay_draws_without_mutating_input() -> None:
    """Verify bounding box overlay returns a new image with green contour drawn."""
    raw = np.zeros((200, 200, 3), dtype=np.uint8)
    box = np.array([[20, 20], [120, 20], [120, 180], [20, 180]], dtype=np.int32)

    out = draw_board_bbox_overlay(raw, box)
    assert out.shape == raw.shape
    assert np.all(raw == 0)  # original unchanged
    assert np.any(out > 0)   # polygon drawn

    # None bbox returns clean copy
    out_none = draw_board_bbox_overlay(raw, None)
    assert np.array_equal(out_none, raw)


def test_draw_camera_roi_overlay_and_status_text() -> None:
    """Verify camera ROI overlay and status text across all 4 state machine states."""
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    trigger = CameraTrigger(roi_relative=(0.2, 0.2, 0.8, 0.8), stable_frames_required=10)

    assert "Đang chờ đặt mạch" in format_trigger_status_text(TriggerState.IDLE_EMPTY)
    assert "7/10" in format_trigger_status_text(
        TriggerState.STABILIZING, {"stable_counter": 7, "required_stable": 10}
    )
    assert "Đã chụp" in format_trigger_status_text(TriggerState.TRIGGER_CAPTURE)
    assert "Nhấc mạch ra" in format_trigger_status_text(TriggerState.COOLDOWN_WAIT_EXIT)

    overlay = draw_camera_roi_overlay(
        frame, trigger, {"state": TriggerState.STABILIZING.value, "stable_counter": 5, "required_stable": 10}
    )
    assert overlay.shape == frame.shape
    assert not np.array_equal(overlay, frame)


def test_format_inspection_badge_html_states() -> None:
    """Verify accessible badge formatting for Normal, Anomaly, and No Board states."""
    res_normal = InspectionResult(
        status="OK",
        is_normal=True,
        anomaly_score=0.14,
        threshold=0.35,
        processing_time_ms=18.5,
    )
    html_norm = format_inspection_badge_html(res_normal)
    assert "TRUE — NORMAL" in html_norm
    assert 'role="status"' in html_norm

    res_anom = InspectionResult(
        status="OK",
        is_normal=False,
        anomaly_score=0.68,
        threshold=0.35,
        processing_time_ms=21.0,
    )
    html_anom = format_inspection_badge_html(res_anom)
    assert "FALSE — ANOMALY" in html_anom

    res_no_board = InspectionResult(
        status="NO_BOARD_DETECTED",
        threshold=0.35,
        message="Không tìm thấy bo mạch",
    )
    html_empty = format_inspection_badge_html(res_no_board)
    assert "NO BOARD DETECTED" in html_empty


def test_recalibrate_history_entries_updates_boolean_on_slider_change() -> None:
    """Verify session history table updates Boolean labels when threshold slider tau changes."""
    history = [
        {"timestamp": "12:00:01", "mode": "Auto-Trigger", "anomaly_score": 0.42, "threshold": 0.30},
        {"timestamp": "12:00:05", "mode": "Manual", "anomaly_score": 0.18, "threshold": 0.30},
    ]

    updated_strict = recalibrate_history_entries(history, new_threshold=0.30)
    assert updated_strict[0]["boolean_result"] is False
    assert updated_strict[0]["result_label"] == "FALSE (Anomaly)"
    assert updated_strict[1]["boolean_result"] is True

    updated_relaxed = recalibrate_history_entries(history, new_threshold=0.50)
    assert updated_relaxed[0]["boolean_result"] is True
    assert updated_relaxed[0]["result_label"] == "TRUE (Normal)"


def test_dataset_helper_save_and_counts(tmp_path: Path) -> None:
    """Verify saving images and counting dataset subfolders for Tab 3 Dataset Helper."""
    counts_init = get_dataset_counts(tmp_path)
    assert counts_init == {"normal_train": 0, "normal_val": 0, "abnormal_test": 0}

    dummy_rgb = np.full((128, 128, 3), 180, dtype=np.uint8)
    saved_train = save_dataset_image(dummy_rgb, "normal_train", dataset_root=tmp_path)
    saved_anom = save_dataset_image(dummy_rgb, "abnormal_test", dataset_root=tmp_path)

    assert saved_train.exists()
    assert saved_anom.exists()

    counts_after = get_dataset_counts(tmp_path)
    assert counts_after == {"normal_train": 1, "normal_val": 0, "abnormal_test": 1}

    with pytest.raises(ValueError):
        save_dataset_image(dummy_rgb, "invalid_subset", dataset_root=tmp_path)
