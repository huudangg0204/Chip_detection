"""Illumination normalization utilities: Gray-World White Balance and CLAHE."""


import cv2
import numpy as np


def gray_world_white_balance(image_rgb: np.ndarray) -> np.ndarray:
    """Apply Gray-World algorithm to correct color cast across varying room lighting.
    
    Args:
        image_rgb: Input RGB image (H, W, 3), uint8.
        
    Returns:
        White-balanced RGB image (H, W, 3), uint8.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return image_rgb

    img_float = image_rgb.astype(np.float32)
    mean_r = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_b = np.mean(img_float[:, :, 2])

    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    scale_r = mean_gray / (mean_r + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_b = mean_gray / (mean_b + 1e-6)

    # Limit maximum scaling to prevent excessive noise/tint amplification
    scale_r = np.clip(scale_r, 0.5, 2.0)
    scale_g = np.clip(scale_g, 0.5, 2.0)
    scale_b = np.clip(scale_b, 0.5, 2.0)

    balanced = np.empty_like(img_float)
    balanced[:, :, 0] = img_float[:, :, 0] * scale_r
    balanced[:, :, 1] = img_float[:, :, 1] * scale_g
    balanced[:, :, 2] = img_float[:, :, 2] * scale_b

    return np.clip(balanced, 0, 255).astype(np.uint8)


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 1.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on L channel of LAB space.
    
    Args:
        image_rgb: Input RGB image (H, W, 3), uint8.
        clip_limit: Threshold for contrast limiting. Default 1.5 to avoid over-amplifying noise.
        tile_grid_size: Grid size for local histogram equalization (default 8x8).
        
    Returns:
        Illumination-normalized RGB image (H, W, 3), uint8.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return image_rgb

    # Convert RGB to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    merged = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def normalize_illumination(
    image_rgb: np.ndarray,
    enable_white_balance: bool = True,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 1.5,
) -> np.ndarray:
    """Pipeline combining Gray-World white balance and CLAHE.
    
    Args:
        image_rgb: Input RGB image (H, W, 3), uint8.
        enable_white_balance: Whether to apply gray-world balance.
        enable_clahe: Whether to apply CLAHE.
        clahe_clip_limit: Clip limit for CLAHE.
        
    Returns:
        Processed RGB image (H, W, 3), uint8.
    """
    out = image_rgb
    if enable_white_balance:
        out = gray_world_white_balance(out)
    if enable_clahe:
        out = apply_clahe(out, clip_limit=clahe_clip_limit)
    return out
