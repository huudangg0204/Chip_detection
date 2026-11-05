"""Data schemas and contracts for Chip Anomaly Detection System."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PinCheckResult:
    """Result of dedicated 3-pin geometric and occlusion inspection."""

    pins_ok: bool
    pin_count: int
    per_pin: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    pin_roi_rgb: np.ndarray | None = None
    overlay_rgb: np.ndarray | None = None


@dataclass
class PreprocessResult:
    """Result of preprocessing pipeline on a single image frame."""

    success: bool
    cropped_rgb: np.ndarray | None = None        # Board image cropped & deskewed (H, W, 3)
    normalized_rgb: np.ndarray | None = None     # Normalized board image (256, 256, 3) for model input
    bbox_box_points: np.ndarray | None = None    # 4 corner points of full_rect on original image (4, 2)
    rotation_angle: float = 0.0                     # Angle rotated in degrees
    error_message: str | None = None             # Error or failure description
    full_board_rgb: np.ndarray | None = None     # Aspect-preserved full board RGB (e.g. 640x144x3)
    board_mask: np.ndarray | None = None         # Clean binary mask (PCB + 3 pins) in cropped_rgb coords
    full_board_mask: np.ndarray | None = None    # Clean binary mask in full_board_rgb coords
    pcb_bbox_points: np.ndarray | None = None    # 4 corner points of bare pcb_rect on original image (4, 2)
    pin_end_margin: float = 0.0                     # Margin from lowest pin tip to bottom of crop / crop height
    warnings: list[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    """Result of full inspection pipeline on a board."""

    status: str                                     # "OK", "NO_BOARD_DETECTED", or "ERROR"
    is_normal: bool | None = None                # True: Normal (board_ok AND pins_ok), False: Anomaly
    anomaly_score: float | None = None           # Anomaly score from PatchCore
    threshold: float = 0.50                         # Decision threshold tau used
    cropped_board_rgb: np.ndarray | None = None  # Cropped board RGB image
    anomaly_map: np.ndarray | None = None        # Heatmap matrix (H, W) float
    heatmap_overlay_rgb: np.ndarray | None = None # Board image blended with Jet heatmap overlay
    processing_time_ms: float = 0.0                 # Total processing time in milliseconds
    message: str = ""                               # Human-readable result summary
    board_ok: bool | None = None                 # True if PCB surface anomaly_score <= threshold
    pins_ok: bool | None = None                  # True if all 3 header pins pass geometric inspection
    pin_result: PinCheckResult | None = None     # Detailed 3-pin inspection result
    warnings: list[str] = field(default_factory=list)
