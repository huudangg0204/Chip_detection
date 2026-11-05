"""Image loader and validator for Phase 1 file uploads and disk access."""

from pathlib import Path

import cv2
import numpy as np


def validate_rgb_image(image: np.ndarray | None) -> bool:
    """Validate that image is a non-empty 3-channel RGB numpy array."""
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if image.ndim != 3 or image.shape[2] != 3:
        return False
    return not (image.shape[0] < 32 or image.shape[1] < 32)


def load_image_from_path(file_path: str | Path) -> np.ndarray:
    """Load image from disk and return as RGB numpy array.
    
    Args:
        file_path: Path to image file.
        
    Returns:
        np.ndarray: RGB image (H, W, 3) uint8.
        
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file cannot be decoded as image.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tệp không tồn tại: {path}")

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Không thể giải mã tệp ảnh: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not validate_rgb_image(rgb):
        raise ValueError(f"Ảnh không hợp lệ hoặc kích thước quá nhỏ: {rgb.shape if rgb is not None else None}")

    return rgb


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes (e.g. from Streamlit file uploader) to RGB numpy array.
    
    Args:
        data: Binary image bytes.
        
    Returns:
        np.ndarray: RGB image (H, W, 3) uint8.
        
    Raises:
        ValueError: If bytes cannot be decoded as image.
    """
    if not data:
        raise ValueError("Dữ liệu bytes rỗng.")

    byte_arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(byte_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Không thể giải mã dữ liệu bytes thành ảnh.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not validate_rgb_image(rgb):
        raise ValueError("Ảnh giải mã không đạt chuẩn RGB hoặc quá nhỏ.")

    return rgb
