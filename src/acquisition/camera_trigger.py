"""Camera ROI management and 4-state Auto-Trigger State Machine for Phase 2."""

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import yaml


class TriggerState(str, Enum):
    IDLE_EMPTY = "IDLE_EMPTY"
    STABILIZING = "STABILIZING"
    TRIGGER_CAPTURE = "TRIGGER_CAPTURE"
    COOLDOWN_WAIT_EXIT = "COOLDOWN_WAIT_EXIT"


class CameraTrigger:
    """State machine that detects when a board is placed, stabilizes, triggers capture once,

    and waits for board removal to prevent duplicate triggers.
    """

    def __init__(
        self,
        roi_relative: tuple[float, float, float, float] = (0.2, 0.2, 0.8, 0.8),
        stable_frames_required: int = 10,
        movement_threshold: float = 5.0,
        cooldown_empty_frames: int = 5,
        min_area_ratio: float = 0.01,
    ) -> None:
        """Initialize camera trigger.

        Args:
            roi_relative: (ymin, xmin, ymax, xmax) normalized to [0.0, 1.0].
            stable_frames_required: Number of consecutive static frames to trigger (default 10).
            movement_threshold: Max center movement (pixels) between frames to consider static.
            cooldown_empty_frames: Consecutive empty frames needed to exit cooldown.
            min_area_ratio: Minimum contour area relative to ROI area to consider board present.
        """
        r0, r1, r2, r3 = (float(roi_relative[0]), float(roi_relative[1]), float(roi_relative[2]), float(roi_relative[3]))
        self.roi_relative: tuple[float, float, float, float] = (r0, r1, r2, r3)
        self.stable_frames_required = int(stable_frames_required)
        self.movement_threshold = float(movement_threshold)
        self.cooldown_empty_frames = int(cooldown_empty_frames)
        self.min_area_ratio = float(min_area_ratio)

        self.state = TriggerState.IDLE_EMPTY
        self.stable_counter = 0
        self.empty_counter = 0
        self.last_center: tuple[float, float] | None = None
        self.candidate_frames: list[tuple[float, np.ndarray]] = []  # (sharpness, frame)

    @classmethod
    def from_config(cls, config_path: str | Path = Path("configs/patchcore_chip.yaml")) -> "CameraTrigger":
        """Create a CameraTrigger instance configured from YAML file."""
        path = Path(config_path)
        cam_cfg: dict[str, object] = {}
        prep_cfg: dict[str, object] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
                cam_cfg = data.get("camera", {}) or {}
                prep_cfg = data.get("preprocess", {}) or {}

        raw_roi = cam_cfg.get("roi", [0.2, 0.2, 0.8, 0.8])
        if isinstance(raw_roi, (list, tuple)) and len(raw_roi) == 4:
            roi_tuple = (float(raw_roi[0]), float(raw_roi[1]), float(raw_roi[2]), float(raw_roi[3]))
        else:
            roi_tuple = (0.2, 0.2, 0.8, 0.8)

        return cls(
            roi_relative=roi_tuple,
            stable_frames_required=int(cam_cfg.get("stable_frames_required", 10)),
            movement_threshold=float(cam_cfg.get("movement_threshold", 5.0)),
            cooldown_empty_frames=int(cam_cfg.get("cooldown_empty_frames", 5)),
            min_area_ratio=float(prep_cfg.get("min_contour_area_ratio", 0.01)),
        )

    def extract_roi(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Crop the ROI region from the camera frame."""
        h, w = frame_rgb.shape[:2]
        ymin, xmin, ymax, xmax = self.roi_relative
        y1, y2 = int(ymin * h), int(ymax * h)
        x1, x2 = int(xmin * w), int(xmax * w)
        return frame_rgb[y1:y2, x1:x2]

    def get_roi_pixel_coords(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) in absolute pixel coordinates for drawing on UI."""
        h, w = frame_shape[:2]
        ymin, xmin, ymax, xmax = self.roi_relative
        return int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

    def _detect_board_in_roi(self, roi_rgb: np.ndarray) -> tuple[bool, tuple[float, float] | None, float]:
        """Detect presence and center of board inside ROI.

        Returns:
            (board_present, center_xy, area)
        """
        roi_h, roi_w = roi_rgb.shape[:2]
        roi_area = float(roi_h * roi_w)

        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return False, None, 0.0

        max_cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(max_cnt))

        if area < self.min_area_ratio * roi_area:
            return False, None, area

        moments = cv2.moments(max_cnt)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            return True, (cx, cy), area

        return False, None, area

    def _calculate_sharpness(self, image_rgb: np.ndarray) -> float:
        """Calculate image sharpness via variance of Laplacian."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def process_frame(
        self, frame_rgb: np.ndarray
    ) -> tuple[TriggerState, bool, np.ndarray | None, dict[str, object]]:
        """Process incoming camera frame through state machine.

        Args:
            frame_rgb: Full RGB frame from webcam.

        Returns:
            Tuple of (current_state, should_trigger, captured_frame, info_dict)
        """
        roi = self.extract_roi(frame_rgb)
        board_present, center, _area = self._detect_board_in_roi(roi)
        should_trigger = False
        captured_frame: np.ndarray | None = None
        movement = 0.0

        # State 1: IDLE_EMPTY
        if self.state == TriggerState.IDLE_EMPTY:
            self.empty_counter = 0
            if board_present and center is not None:
                self.state = TriggerState.STABILIZING
                self.stable_counter = 1
                self.last_center = center
                sharpness = self._calculate_sharpness(roi)
                self.candidate_frames = [(sharpness, frame_rgb.copy())]
            else:
                self.stable_counter = 0
                self.last_center = None
                self.candidate_frames = []

        # State 2: STABILIZING
        elif self.state == TriggerState.STABILIZING:
            if not board_present or center is None or self.last_center is None:
                # Lost board while stabilizing -> back to IDLE
                self.state = TriggerState.IDLE_EMPTY
                self.stable_counter = 0
                self.last_center = None
                self.candidate_frames = []
            else:
                movement = float(np.hypot(center[0] - self.last_center[0], center[1] - self.last_center[1]))
                self.last_center = center

                if movement < self.movement_threshold:
                    self.stable_counter += 1
                    sharpness = self._calculate_sharpness(roi)
                    self.candidate_frames.append((sharpness, frame_rgb.copy()))

                    if self.stable_counter >= self.stable_frames_required:
                        # Mạch đã đứng yên đủ số frames -> Trigger!
                        self.state = TriggerState.TRIGGER_CAPTURE
                        should_trigger = True
                        # Select sharpest frame among candidates
                        best_candidate = max(self.candidate_frames, key=lambda x: x[0])
                        captured_frame = best_candidate[1]
                        self.candidate_frames = []
                else:
                    # Mạch còn đang di chuyển -> reset bộ đếm ổn định
                    self.stable_counter = 1
                    sharpness = self._calculate_sharpness(roi)
                    self.candidate_frames = [(sharpness, frame_rgb.copy())]

        # State 3: TRIGGER_CAPTURE -> auto transition to COOLDOWN_WAIT_EXIT
        elif self.state == TriggerState.TRIGGER_CAPTURE:
            self.state = TriggerState.COOLDOWN_WAIT_EXIT
            self.empty_counter = 0

        # State 4: COOLDOWN_WAIT_EXIT
        elif self.state == TriggerState.COOLDOWN_WAIT_EXIT:
            if not board_present:
                self.empty_counter += 1
                if self.empty_counter >= self.cooldown_empty_frames:
                    # Người dùng đã lấy mạch ra khỏi ROI -> trở lại IDLE_EMPTY
                    self.state = TriggerState.IDLE_EMPTY
                    self.empty_counter = 0
                    self.stable_counter = 0
                    self.last_center = None
            else:
                self.empty_counter = 0

        info = {
            "state": self.state.value,
            "board_present": board_present,
            "stable_counter": self.stable_counter,
            "required_stable": self.stable_frames_required,
            "movement": round(movement, 2),
            "empty_counter": self.empty_counter,
        }

        return self.state, should_trigger, captured_frame, info

    def force_manual_trigger(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Trigger capture manually (bypassing state machine) for manual inspection mode."""
        self.state = TriggerState.COOLDOWN_WAIT_EXIT
        self.empty_counter = 0
        self.stable_counter = 0
        return frame_rgb.copy()

    def reset(self) -> None:
        """Reset state machine to initial empty state."""
        self.state = TriggerState.IDLE_EMPTY
        self.stable_counter = 0
        self.empty_counter = 0
        self.last_center = None
        self.candidate_frames = []
