"""Unit tests for CameraTrigger state machine and debounce mechanism."""

import numpy as np

from src.acquisition.camera_trigger import CameraTrigger, TriggerState


def create_synthetic_frame_with_board(board_center_x: int = 320, board_center_y: int = 240) -> np.ndarray:
    """Create a 640x480 frame with white background and a dark rectangular board at specified center."""
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    bw, bh = 40, 160
    x1 = max(board_center_x - bw // 2, 0)
    x2 = min(board_center_x + bw // 2, 640)
    y1 = max(board_center_y - bh // 2, 0)
    y2 = min(board_center_y + bh // 2, 480)
    frame[y1:y2, x1:x2] = [20, 25, 20]  # dark board
    return frame


def create_empty_frame() -> np.ndarray:
    """Create a blank white frame without any board."""
    return np.full((480, 640, 3), 245, dtype=np.uint8)


def test_initial_state() -> None:
    """Trigger should start in IDLE_EMPTY state."""
    trigger = CameraTrigger(stable_frames_required=5)
    assert trigger.state == TriggerState.IDLE_EMPTY


def test_empty_frames_stay_idle() -> None:
    """Processing empty frames should keep state in IDLE_EMPTY with no triggers."""
    trigger = CameraTrigger(stable_frames_required=5)
    empty = create_empty_frame()

    for _ in range(10):
        state, should_trigger, captured, info = trigger.process_frame(empty)
        assert state == TriggerState.IDLE_EMPTY
        assert should_trigger is False
        assert captured is None
        assert info["board_present"] is False


def test_stable_sequence_triggers_once() -> None:
    """A board held stationary for N frames should trigger exactly once and enter cooldown."""
    required = 10
    trigger = CameraTrigger(
        roi_relative=(0.1, 0.1, 0.9, 0.9),
        stable_frames_required=required,
        movement_threshold=5.0,
        cooldown_empty_frames=3,
    )

    board_frame = create_synthetic_frame_with_board(320, 240)
    trigger_count = 0

    # Feed stationary board frames
    for i in range(required + 5):
        state, should_trigger, captured, _ = trigger.process_frame(board_frame)
        if should_trigger:
            trigger_count += 1
            assert captured is not None
            assert state == TriggerState.TRIGGER_CAPTURE

    # Must trigger exactly once!
    assert trigger_count == 1, f"Expected exactly 1 trigger, got {trigger_count}"
    assert trigger.state in (TriggerState.TRIGGER_CAPTURE, TriggerState.COOLDOWN_WAIT_EXIT)


def test_movement_resets_stability_counter() -> None:
    """Moving board should reset the stable_counter and prevent premature trigger."""
    required = 6
    trigger = CameraTrigger(
        roi_relative=(0.1, 0.1, 0.9, 0.9),
        stable_frames_required=required,
        movement_threshold=5.0,
    )

    # Frame 1: place board at (300, 240)
    state, trig, _, info = trigger.process_frame(create_synthetic_frame_with_board(300, 240))
    assert state == TriggerState.STABILIZING
    assert trig is False

    # Move board by 20 pixels each frame (above threshold 5.0)
    for shift in range(1, 10):
        pos_x = 300 + shift * 20
        state, trig, _, info = trigger.process_frame(create_synthetic_frame_with_board(pos_x, 240))
        assert state == TriggerState.STABILIZING
        assert trig is False
        assert info["stable_counter"] == 1  # reset continuously


def test_cooldown_and_exit() -> None:
    """After trigger, must stay in cooldown until board is removed for cooldown_empty_frames."""
    trigger = CameraTrigger(
        roi_relative=(0.1, 0.1, 0.9, 0.9),
        stable_frames_required=3,
        cooldown_empty_frames=3,
    )
    board_frame = create_synthetic_frame_with_board(320, 240)

    # Transition to TRIGGER_CAPTURE
    for _ in range(3):
        trigger.process_frame(board_frame)

    # Advance one more frame while board remains in ROI -> must be in COOLDOWN
    state, trig, _, _ = trigger.process_frame(board_frame)
    assert state == TriggerState.COOLDOWN_WAIT_EXIT
    assert trig is False

    # Feed 2 empty frames -> still in cooldown (< 3)
    empty = create_empty_frame()
    for _ in range(2):
        state, trig, _, _ = trigger.process_frame(empty)
        assert state == TriggerState.COOLDOWN_WAIT_EXIT

    # 3rd empty frame -> back to IDLE_EMPTY
    state, trig, _, _ = trigger.process_frame(empty)
    assert state == TriggerState.IDLE_EMPTY


def test_manual_trigger_and_reset() -> None:
    """Test manual trigger and state machine reset."""
    trigger = CameraTrigger(stable_frames_required=5)
    frame = create_synthetic_frame_with_board(320, 240)

    captured = trigger.force_manual_trigger(frame)
    assert captured.shape == frame.shape
    assert trigger.state == TriggerState.COOLDOWN_WAIT_EXIT

    trigger.reset()
    assert trigger.state == TriggerState.IDLE_EMPTY
    assert trigger.stable_counter == 0


def test_camera_trigger_from_config() -> None:
    """Verify CameraTrigger.from_config loads consistent ROI and threshold from YAML."""
    trigger = CameraTrigger.from_config("configs/patchcore_chip.yaml")
    assert trigger.roi_relative == (0.2, 0.2, 0.8, 0.8)
    assert trigger.stable_frames_required == 10
    assert trigger.movement_threshold == 5.0
    assert trigger.cooldown_empty_frames == 5
    x1, y1, x2, y2 = trigger.get_roi_pixel_coords((480, 640, 3))
    assert (x1, y1, x2, y2) == (128, 96, 512, 384)
