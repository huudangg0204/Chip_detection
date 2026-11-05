"""Unit tests for BoardAligner and illumination preprocessing pipeline."""

from pathlib import Path

import numpy as np
import pytest

from src.acquisition.image_loader import load_image_from_path
from src.preprocessing.board_aligner import BoardAligner, create_board_aligner, preprocess_board_image
from src.preprocessing.illumination import apply_clahe, gray_world_white_balance, normalize_illumination


@pytest.fixture
def sample1_rgb() -> np.ndarray:
    path = Path("image/sample1.jpg")
    assert path.exists(), f"Sample image {path} does not exist"
    return load_image_from_path(path)


@pytest.fixture
def sample2_rgb() -> np.ndarray:
    path = Path("image/sample2.jpg")
    assert path.exists(), f"Sample image {path} does not exist"
    return load_image_from_path(path)


@pytest.fixture
def aligner() -> BoardAligner:
    return BoardAligner(target_size=(256, 256), apply_clahe_flag=True, resize_mode="pad")


def test_illumination_functions() -> None:
    """Test white balance and CLAHE on synthetic inputs."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    wb = gray_world_white_balance(img)
    assert wb.shape == (100, 100, 3)
    assert wb.dtype == np.uint8

    clahe = apply_clahe(img, clip_limit=1.5)
    assert clahe.shape == (100, 100, 3)
    assert clahe.dtype == np.uint8

    norm = normalize_illumination(img)
    assert norm.shape == (100, 100, 3)


def test_board_aligner_sample1(sample1_rgb: np.ndarray, aligner: BoardAligner) -> None:
    """Test board alignment on sample1.jpg (anomaly sample)."""
    result = aligner.process(sample1_rgb)
    assert result.success is True, f"Failed on sample1: {result.error_message}"
    assert result.cropped_rgb is not None
    assert result.normalized_rgb is not None

    # Check orientation: board must be upright (height > width)
    h, w = result.cropped_rgb.shape[:2]
    assert h > w, f"Expected vertical board (h > w), got h={h}, w={w}"

    # Check normalized dimensions match target_size
    assert result.normalized_rgb.shape == (256, 256, 3)

    # Check bounding box points exist
    assert result.bbox_box_points is not None
    assert result.bbox_box_points.shape == (4, 2)

    import cv2

    Path("outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("outputs/debug_sample1.jpg", cv2.cvtColor(result.normalized_rgb, cv2.COLOR_RGB2BGR))


def test_board_aligner_sample2(sample2_rgb: np.ndarray, aligner: BoardAligner) -> None:
    """Test board alignment on sample2.jpg (normal sample)."""
    result = aligner.process(sample2_rgb)
    assert result.success is True, f"Failed on sample2: {result.error_message}"
    assert result.cropped_rgb is not None
    assert result.normalized_rgb is not None

    # Check orientation: board must be upright (height > width)
    h, w = result.cropped_rgb.shape[:2]
    assert h > w, f"Expected vertical board (h > w), got h={h}, w={w}"

    # Check normalized dimensions match target_size
    assert result.normalized_rgb.shape == (256, 256, 3)
    assert result.bbox_box_points is not None

    import cv2

    Path("outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("outputs/debug_sample2.jpg", cv2.cvtColor(result.normalized_rgb, cv2.COLOR_RGB2BGR))


def test_board_aligner_blank_image(aligner: BoardAligner) -> None:
    """Test that a blank white sheet returns success=False without crashing."""
    blank_white = np.full((1000, 800, 3), 250, dtype=np.uint8)
    result = aligner.process(blank_white)
    assert result.success is False
    assert result.cropped_rgb is None
    assert result.normalized_rgb is None
    assert "Không tìm thấy bo mạch" in (result.error_message or "")


def test_board_aligner_empty_input(aligner: BoardAligner) -> None:
    """Test handling of empty or None inputs."""
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    result = aligner.process(empty)
    assert result.success is False
    assert "không hợp lệ" in (result.error_message or "")


def test_board_aligner_spec_signature_and_margin(sample2_rgb: np.ndarray) -> None:
    """Verify Spec signature BoardAligner(target_size=(256, 256), apply_clahe=True) and margin expansion."""
    aligner_no_margin = BoardAligner(target_size=(256, 256), apply_clahe=True, crop_margin_ratio=0.0)
    aligner_with_margin = BoardAligner(target_size=(256, 256), apply_clahe=True, crop_margin_ratio=0.04)

    res_tight = aligner_no_margin.process(sample2_rgb)
    res_padded = aligner_with_margin.process(sample2_rgb)

    assert res_tight.success is True
    assert res_padded.success is True
    assert res_tight.cropped_rgb is not None
    assert res_padded.cropped_rgb is not None
    assert res_padded.cropped_rgb.shape[0] > res_tight.cropped_rgb.shape[0]
    assert res_padded.cropped_rgb.shape[1] > res_tight.cropped_rgb.shape[1]


def test_board_aligner_180_degree_canonicalization(sample2_rgb: np.ndarray, aligner: BoardAligner) -> None:
    """Verify a board rotated 180 degrees upside down is automatically canonicalized to upright orientation."""
    import cv2

    upside_down = cv2.rotate(sample2_rgb, cv2.ROTATE_180)
    res_normal = aligner.process(sample2_rgb)
    res_flipped = aligner.process(upside_down)

    assert res_normal.success is True
    assert res_flipped.success is True
    assert res_normal.normalized_rgb is not None
    assert res_flipped.normalized_rgb is not None

    # After 180-deg canonicalization, both images must have matching upright structure
    diff = float(np.mean(np.abs(res_normal.normalized_rgb.astype(np.float32) - res_flipped.normalized_rgb.astype(np.float32))))
    assert diff < 8.0, f"180-deg canonicalized board differs too much from upright board: mean diff={diff}"


def test_board_aligner_structural_similarity_between_samples(
    sample1_rgb: np.ndarray, sample2_rgb: np.ndarray, aligner: BoardAligner
) -> None:
    """Verify aligned sample1 and sample2 have high structural similarity (SSIM/NCC) on non-defective board zones."""
    res1 = aligner.process(sample1_rgb)
    res2 = aligner.process(sample2_rgb)
    assert res1.normalized_rgb is not None
    assert res2.normalized_rgb is not None

    # Compare the upper-middle non-defective region (LEDs + mounting hole + potentiometer, y=25..135, x=96..160)
    patch1 = res1.normalized_rgb[25:135, 96:160].astype(np.float32)
    patch2 = res2.normalized_rgb[25:135, 96:160].astype(np.float32)

    mu1, mu2 = float(np.mean(patch1)), float(np.mean(patch2))
    sigma1, sigma2 = float(np.var(patch1)), float(np.var(patch2))
    sigma12 = float(np.mean((patch1 - mu1) * (patch2 - mu2)))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    assert ssim > 0.65, f"Expected high structural similarity on non-defective zone, got SSIM={ssim:.3f}"


def test_is_upside_down_robust_to_missing_pins_and_lighting_gradient(
    sample2_rgb: np.ndarray, aligner: BoardAligner
) -> None:
    """Verify is_upside_down correctly detects orientation even when bottom pins are cropped off or top lighting is bright."""
    import cv2

    res = aligner.process(sample2_rgb)
    assert res.success is True
    assert res.cropped_rgb is not None
    crop = res.cropped_rgb
    h = crop.shape[0]

    # Case 1: Protruding bottom pins cropped off (only dark PCB body remains at the tail)
    pcb_only = crop[: int(h * 0.80), :]
    assert aligner.is_upside_down(pcb_only) is False
    assert aligner.is_upside_down(cv2.rotate(pcb_only, cv2.ROTATE_180)) is True

    # Case 2: Strong top-to-bottom illumination gradient (+40 at top, -40 at bottom)
    grad = np.linspace(40.0, -40.0, h, dtype=np.float32)[:, None, None]
    top_lit = np.clip(crop.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    assert aligner.is_upside_down(top_lit) is False
    assert aligner.is_upside_down(cv2.rotate(top_lit, cv2.ROTATE_180)) is True


def test_shared_preprocess_board_image_rotations_and_uneven_lighting(sample2_rgb: np.ndarray) -> None:
    """Verify shared preprocess_board_image handles 90/180/270 rotations and uneven paper shading."""
    import cv2

    shared_aligner = create_board_aligner()
    res_0 = preprocess_board_image(sample2_rgb, aligner=shared_aligner)
    assert res_0.success is True
    assert res_0.normalized_rgb is not None

    for rot_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated_img = cv2.rotate(sample2_rgb, rot_code)
        res_rot = preprocess_board_image(rotated_img, aligner=shared_aligner)
        assert res_rot.success is True
        assert res_rot.normalized_rgb is not None
        diff = float(
            np.mean(np.abs(res_0.normalized_rgb.astype(np.float32) - res_rot.normalized_rgb.astype(np.float32)))
        )
        assert diff < 10.0, f"Rotation {rot_code} failed upright canonicalization (diff={diff:.2f})"

    shaded_path = Path("data/chip_dataset/normal/train/normal_train_007.jpg")
    if shaded_path.exists():
        shaded_rgb = load_image_from_path(shaded_path)
        res_shaded = preprocess_board_image(shaded_rgb, aligner=shared_aligner)
        assert res_shaded.success is True
        assert res_shaded.cropped_rgb is not None
        assert res_shaded.normalized_rgb is not None
        assert shared_aligner.is_upside_down(res_shaded.cropped_rgb) is False


def test_full_board_pin_extension_and_pin_end_margin(sample2_rgb: np.ndarray) -> None:
    """Verify full_rect extends beyond pcb_rect to include all 3 pins with >= 3% bottom margin."""
    aligner = create_board_aligner()
    res = aligner.process(sample2_rgb)
    assert res.success is True
    assert res.full_board_rgb is not None
    assert res.full_board_rgb.shape == (640, 144, 3)
    assert res.pcb_bbox_points is not None
    assert res.bbox_box_points is not None
    assert res.pcb_bbox_points.shape == (4, 2)
    assert res.bbox_box_points.shape == (4, 2)
    assert res.pin_end_margin >= 0.03, f"Expected >= 3% pin end margin, got {res.pin_end_margin:.4f}"


def test_near_frame_edge_warning_and_padding(sample2_rgb: np.ndarray) -> None:
    """Verify board near image bottom edge triggers 'board near frame edge' warning and pads cleanly."""
    aligner = create_board_aligner()
    res_base = aligner.process(sample2_rgb)
    assert res_base.success is True
    assert res_base.pcb_bbox_points is not None

    # Crop the raw image tightly just below the PCB bottom edge so pin extension goes out of frame
    max_pcb_y = int(np.max(res_base.pcb_bbox_points[:, 1]))
    tight_raw = sample2_rgb[: min(sample2_rgb.shape[0], max_pcb_y + 5), :, :].copy()

    res_edge = aligner.process(tight_raw)
    assert res_edge.success is True
    assert res_edge.full_board_rgb is not None
    assert "board near frame edge" in res_edge.warnings


def test_flat_field_multi_otsu_shadow_resistance_and_background_masking(sample2_rgb: np.ndarray) -> None:
    """Verify artificial directional shadow (30% darkening on right side) is excluded from board_mask and masked out."""
    aligner = create_board_aligner()
    res_clean = aligner.process(sample2_rgb)
    assert res_clean.success is True
    assert res_clean.board_mask is not None
    assert res_clean.normalized_rgb is not None

    # Create synthetic directional cast shadow on the right side of the board (~32% darker paper)
    _h, w = sample2_rgb.shape[:2]
    shadowed = sample2_rgb.astype(np.float32).copy()
    x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    # Darken right half by up to 32%
    shadow_factor = np.where(x_coords > 0.45, 1.0 - 0.32 * np.clip((x_coords - 0.45) / 0.25, 0.0, 1.0), 1.0)
    shadowed = np.clip(shadowed * shadow_factor, 0, 255).astype(np.uint8)

    res_shadow = aligner.process(shadowed)
    assert res_shadow.success is True
    assert res_shadow.board_mask is not None
    assert res_shadow.normalized_rgb is not None

    # 1. Pixels outside board_mask must be strictly zeroed [0, 0, 0]
    outside_pixels = res_shadow.normalized_rgb[res_shadow.board_mask == 0]
    assert outside_pixels.size > 0
    assert int(np.max(outside_pixels)) == 0

    # 2. Board mask area should not bloat due to the right-side cast shadow
    clean_area = float(np.mean(res_clean.board_mask > 0))
    shadow_area = float(np.mean(res_shadow.board_mask > 0))
    assert abs(shadow_area - clean_area) < 0.04, (
        f"Shadow caused board_mask area drift: clean={clean_area:.3f}, shadow={shadow_area:.3f}"
    )


