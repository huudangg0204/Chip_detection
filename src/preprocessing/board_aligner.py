"""Board alignment, shadow-resistant segmentation, and deskew pipeline for LM393 chip inspection."""

import argparse
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.preprocessing.illumination import normalize_illumination
from src.schemas import PreprocessResult

logger = logging.getLogger(__name__)


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points in clockwise order: [top-left, top-right, bottom-right, bottom-left].

    Args:
        pts: Array of shape (4, 2)

    Returns:
        Ordered array of shape (4, 2), float32.
    """
    pts_f = np.asarray(pts, dtype=np.float32)
    center = np.mean(pts_f, axis=0)
    angles = np.arctan2(pts_f[:, 1] - center[1], pts_f[:, 0] - center[0])
    cw_pts = pts_f[np.argsort(angles)]

    # Roll so that top-left (smallest x + y) is at index 0
    tl_idx = int(np.argmin(cw_pts.sum(axis=1)))
    return np.roll(cw_pts, -tl_idx, axis=0).astype(np.float32)


def canonicalize_rect_corners(box: np.ndarray) -> np.ndarray:
    """Order 4 corners [TL, TR, BR, BL] clockwise such that TL->TR is short edge (width) and TL->BL is long edge."""
    ordered = order_box_points(box)
    edge1 = np.linalg.norm(ordered[0] - ordered[1])
    edge2 = np.linalg.norm(ordered[0] - ordered[3])
    if edge1 > edge2:
        ordered = np.roll(ordered, 1, axis=0)
    return ordered.astype(np.float32)


def multi_otsu_3class_u8(img_u8: np.ndarray) -> tuple[int, int, int]:
    """Compute 2-class Otsu threshold and 3-class Multi-Otsu thresholds (t1, t2) on a uint8 image.

    Separates 3 illumination classes: Dark PCB (< t1), Cast Shadow (t1..t2), and Paper Background (> t2).

    Returns:
        Tuple of (t_otsu2, t1_dark_pcb, t2_shadow_bg).
    """
    hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 100, 90, 170

    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256, dtype=np.float64))
    mu_t = float(mu[-1])

    denom2 = omega * (1.0 - omega)
    valid2 = denom2 > 1e-9
    sigma2 = np.zeros(256, dtype=np.float64)
    sigma2[valid2] = ((mu_t * omega[valid2] - mu[valid2]) ** 2) / denom2[valid2]
    t_otsu2 = int(np.argmax(sigma2))

    idx = np.arange(15, 240)
    w0 = omega[idx][:, None]
    w2 = 1.0 - omega[idx][None, :]
    w1 = omega[idx][None, :] - omega[idx][:, None]

    m0 = mu[idx][:, None]
    m2 = mu_t - mu[idx][None, :]
    m1 = mu[idx][None, :] - mu[idx][:, None]

    valid3 = (w0 > 1e-6) & (w1 > 1e-6) & (w2 > 1e-6)
    tri = np.triu(np.ones_like(valid3, dtype=bool), k=10)
    valid3 = valid3 & tri

    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 1e-6)
    mu1 = np.divide(m1, w1, out=np.zeros_like(m1), where=w1 > 1e-6)
    mu2 = np.divide(m2, w2, out=np.zeros_like(m2), where=w2 > 1e-6)
    raw_score3 = (
        w0 * (mu0 - mu_t) ** 2
        + w1 * (mu1 - mu_t) ** 2
        + w2 * (mu2 - mu_t) ** 2
    )
    score3 = np.where(valid3, raw_score3, 0.0)
    i1, i2 = np.unravel_index(np.argmax(score3), score3.shape)
    t1, t2 = int(idx[i1]), int(idx[i2])
    return t_otsu2, t1, t2


def estimate_flat_field_ratio(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Estimate smooth background illumination on LAB L-channel and compute flat-field ratio.

    Returns:
        Tuple of (ratio_u8 in [0..255], bg_l float32, estimated_bg_rgb tuple).
    """
    h, w = image_rgb.shape[:2]
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float32)

    sw = 160
    sh = max(60, round(h * sw / max(w, 1)))
    l_small = cv2.resize(l_ch, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(35, int(min(sw, sh) * 0.45)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_small = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (k, k), 0)
    bg_l = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

    ratio = np.clip(l_ch / np.maximum(bg_l, 25.0), 0.0, 1.15)
    ratio_u8 = np.clip(ratio * 255.0, 0, 255).astype(np.uint8)

    bg_mask = ratio_u8 > 210
    if np.count_nonzero(bg_mask) > 100:
        med_rgb = np.median(image_rgb[bg_mask], axis=0)
        bg_rgb = (int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2]))
    else:
        med_rgb = np.percentile(image_rgb.reshape(-1, 3), 85, axis=0)
        bg_rgb = (int(med_rgb[0]), int(med_rgb[1]), int(med_rgb[2]))

    return ratio_u8, bg_l, bg_rgb


class BoardAligner:
    """Detects, deskews, canonicalizes (180 deg), masks shadows, and normalizes chip boards."""

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 1.5,
        crop_margin_ratio: float = 0.025,
        min_area_ratio: float = 0.008,
        max_area_ratio: float = 0.70,
        min_aspect_ratio: float = 2.0,
        max_aspect_ratio: float = 6.0,
        resize_mode: str = "pad",  # "pad" or "stretch"
        *,
        apply_clahe_flag: bool | None = None,
        enable_white_balance: bool = True,
        full_board_size: tuple[int, int] = (144, 640),  # (width, height) aspect-preserved
        pin_extension_ratio: float = 0.22,
        head_margin_ratio: float | None = None,
        side_margin_ratio: float | None = None,
        pcb_aspect_ratio_expected: float = 4.0,
        pcb_aspect_ratio_tolerance: float = 0.65,
        mask_background: bool = True,
        background_fill_rgb: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        resolved_clahe = apply_clahe if apply_clahe_flag is None else apply_clahe_flag
        self.target_size = target_size
        self.full_board_size = full_board_size
        self.apply_clahe = bool(resolved_clahe)
        self.apply_clahe_flag = bool(resolved_clahe)
        self.enable_white_balance = bool(enable_white_balance)
        self.clahe_clip_limit = clahe_clip_limit
        self.crop_margin_ratio = crop_margin_ratio
        self.head_margin_ratio = crop_margin_ratio if head_margin_ratio is None else float(head_margin_ratio)
        self.side_margin_ratio = crop_margin_ratio if side_margin_ratio is None else float(side_margin_ratio)
        self.pin_extension_ratio = float(pin_extension_ratio)
        self.pcb_aspect_ratio_expected = float(pcb_aspect_ratio_expected)
        self.pcb_aspect_ratio_tolerance = float(pcb_aspect_ratio_tolerance)
        self.mask_background = bool(mask_background)
        self.background_fill_rgb = background_fill_rgb
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.resize_mode = resize_mode

    def _clean_pcb_mask_from_thresh(self, raw_bin: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        """Close component gaps, fill internal holes (mounting holes, 103 pot), and open away thin shadow wisps."""
        scale = max(0.5, min(img_h, img_w) / 800.0)
        k_close = max(3, round(7 * scale)) | 1
        k_open = max(3, round(9 * scale)) | 1

        ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        closed = cv2.morphologyEx(raw_bin, cv2.MORPH_CLOSE, ker_close)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        img_area = float(img_h * img_w)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 0.004 * img_area:
                cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)

        ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ker_open)
        return opened

    def _find_best_contour_in_mask(
        self,
        closed_mask: np.ndarray,
        img_area: float,
    ) -> np.ndarray | None:
        """Select the largest contour matching board area and aspect ratio constraints."""
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_cnt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area_ratio * img_area <= area <= self.max_area_ratio * img_area):
                continue

            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w <= 0 or h <= 0:
                continue

            ar = max(w, h) / min(w, h)
            if self.min_aspect_ratio <= ar <= self.max_aspect_ratio and area > best_area:
                best_area = area
                best_cnt = cnt

        return best_cnt

    def _segment_board_with_flat_field(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, tuple[int, int, int]]:
        """Segment bare dark PCB body using flat-field illumination ratio and 3-class Multi-Otsu."""
        img_h, img_w = image_rgb.shape[:2]
        img_area = float(img_h * img_w)

        ratio_u8, _, bg_rgb = estimate_flat_field_ratio(image_rgb)
        if float(np.std(ratio_u8)) < 4.0:
            return None, None, ratio_u8, bg_rgb

        blur_r = cv2.GaussianBlur(ratio_u8, (5, 5), 0)
        t_otsu2, t1, _ = multi_otsu_3class_u8(blur_r)

        # Candidate dark-PCB thresholds in flat-field ratio space:
        # Cast shadows have ratio_u8 in [120..215] (0.47..0.84), while black PCB has ratio_u8 < 115.
        candidates = [
            int(np.clip(t1, 75, 115)),
            int(np.clip(min(t1, int(t_otsu2 * 0.72)), 70, 110)),
            95,
            110,
            int(np.clip(t_otsu2, 80, 135)),
        ]

        last_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for thr in candidates:
            _, raw_bin = cv2.threshold(blur_r, thr, 255, cv2.THRESH_BINARY_INV)
            cleaned = self._clean_pcb_mask_from_thresh(raw_bin, img_h, img_w)
            last_mask = cleaned
            best_cnt = self._find_best_contour_in_mask(cleaned, img_area)
            if best_cnt is not None:
                single_mask = np.zeros_like(cleaned)
                cv2.drawContours(single_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
                return best_cnt, single_mask, ratio_u8, bg_rgb

        return None, last_mask, ratio_u8, bg_rgb

    def segment_board(self, image_rgb: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Segment dark PCB contour from background and cast shadows using flat-field + Multi-Otsu.

        Args:
            image_rgb: RGB image (H, W, 3).

        Returns:
            Tuple of (best_contour, binary_pcb_mask) or (None, None) if not found.
        """
        best_cnt, pcb_mask, _, _ = self._segment_board_with_flat_field(image_rgb)
        return best_cnt, pcb_mask

    @staticmethod
    def _find_pcb_x_bounds(gray: np.ndarray) -> tuple[int, int]:
        """Find horizontal [x0, x1] bounds of the dark PCB body, ignoring side paper/shadows."""
        h, w = gray.shape[:2]
        if w < 10 or h < 10:
            return 0, w
        mid_band = gray[int(h * 0.15) : max(int(h * 0.15) + 1, int(h * 0.75)), :]
        col_darkness = np.percentile(mid_band, 30, axis=0)
        lo, hi = float(np.min(col_darkness)), float(np.max(col_darkness))
        if hi - lo < 15.0:
            return 0, w
        thresh = lo + 0.45 * (hi - lo)
        pcb_cols = np.where(col_darkness <= thresh)[0]
        if pcb_cols.size < max(4, int(w * 0.25)):
            return 0, w
        return int(pcb_cols[0]), int(pcb_cols[-1] + 1)

    @staticmethod
    def _peak_hole_contrast(
        g: np.ndarray,
        y_start: int,
        y_end: int,
        win_h: int,
        c_a: int,
        c_b: int,
        l_a: int,
        l_b: int,
        r_a: int,
        r_b: int,
    ) -> float:
        """Compute peak (center - shoulders) contrast over a sliding vertical window."""
        if y_end <= y_start + 1:
            return 0.0
        band_c = np.mean(g[y_start:y_end, c_a:c_b], axis=1)
        band_l = np.mean(g[y_start:y_end, l_a:l_b], axis=1)
        band_r = np.mean(g[y_start:y_end, r_a:r_b], axis=1)
        row_contrast = band_c - 0.5 * (band_l + band_r)
        k = max(1, min(win_h, row_contrast.size))
        kernel = np.ones(k, dtype=np.float32) / float(k)
        smoothed = np.convolve(row_contrast, kernel, mode="valid")
        return float(np.max(smoothed))

    def is_upside_down(self, aligned_rgb: np.ndarray) -> bool:
        """Determine if aligned board is upside down (180 deg reversed)."""
        h, w = aligned_rgb.shape[:2]
        if h < 16 or w < 8:
            return False

        gray = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        x0, x1 = self._find_pcb_x_bounds(gray)
        pcb_gray = gray[:, x0:x1]
        pcb_rgb = aligned_rgb[:, x0:x1]
        pw = pcb_gray.shape[1]

        c_a, c_b = max(0, int(pw * 0.34)), min(pw, max(1, int(pw * 0.66)))
        l_a, l_b = max(0, int(pw * 0.08)), min(pw, max(1, int(pw * 0.28)))
        r_a, r_b = max(0, int(pw * 0.72)), min(pw, max(1, int(pw * 0.92)))
        win_h = max(2, int(h * 0.05))

        def _head_likelihood(g: np.ndarray, rgb_arr: np.ndarray) -> float:
            hole1 = self._peak_hole_contrast(
                g,
                max(1, int(h * 0.02)),
                max(3, int(h * 0.18)),
                win_h,
                c_a,
                c_b,
                l_a,
                l_b,
                r_a,
                r_b,
            )
            y2_a, y2_b = max(1, int(h * 0.25)), max(2, int(h * 0.37))
            h2_c = float(np.mean(g[y2_a:y2_b, c_a:c_b]))
            h2_s = 0.5 * (
                float(np.mean(g[y2_a:y2_b, l_a:l_b]))
                + float(np.mean(g[y2_a:y2_b, r_a:r_b]))
            )
            hole2 = h2_c - h2_s

            yb_a, yb_b = max(1, int(h * 0.12)), max(2, int(h * 0.48))
            b_ch = rgb_arr[yb_a:yb_b, :, 2].astype(np.float32)
            r_ch = rgb_arr[yb_a:yb_b, :, 0].astype(np.float32)
            blue_score = float(np.mean(np.maximum(b_ch - r_ch, 0.0)))

            return 1.2 * hole1 + 0.6 * hole2 + 1.8 * blue_score

        top_head = _head_likelihood(pcb_gray, pcb_rgb)
        bot_head = _head_likelihood(
            cv2.rotate(pcb_gray, cv2.ROTATE_180),
            cv2.rotate(pcb_rgb, cv2.ROTATE_180),
        )

        if abs(top_head - bot_head) >= 5.0:
            return bool(bot_head > top_head)

        top_h = max(int(h * 0.25), 1)
        bot_h = max(int(h * 0.25), 1)
        top_gray = gray[:top_h, :]
        bot_gray = gray[h - bot_h :, :]
        top_score = float(np.mean(top_gray)) + float(
            np.mean(np.abs(cv2.Sobel(top_gray, cv2.CV_64F, 1, 0, ksize=3)))
        )
        bot_score = float(np.mean(bot_gray)) + float(
            np.mean(np.abs(cv2.Sobel(bot_gray, cv2.CV_64F, 1, 0, ksize=3)))
        )
        return bool(top_score > bot_score)

    @staticmethod
    def _warp_from_corners(
        image: np.ndarray,
        corners: np.ndarray,
        out_w: int,
        out_h: int,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: tuple[int, ...] | int = 0,
    ) -> np.ndarray:
        """Perspective-warp a 4-corner quad [TL, TR, BR, BL] into (out_h, out_w)."""
        dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return cv2.warpPerspective(
            image,
            M,
            (out_w, out_h),
            borderMode=border_mode,
            borderValue=border_value,
        )

    @staticmethod
    def _build_crop_board_mask_and_pin_margin(
        full_crop_rgb: np.ndarray,
        x_pcb_left: int,
        x_pcb_right: int,
        y_pcb_top: int,
        y_pcb_bot: int,
    ) -> tuple[np.ndarray, float]:
        """Construct clean board_mask (PCB + 3 pins) in crop coordinates and measure pin_end_margin."""
        full_h, full_w = full_crop_rgb.shape[:2]
        board_mask = np.zeros((full_h, full_w), dtype=np.uint8)

        xl = int(np.clip(x_pcb_left, 0, full_w - 1))
        xr = int(np.clip(x_pcb_right, xl + 1, full_w))
        yt = int(np.clip(y_pcb_top, 0, full_h - 1))
        yb = int(np.clip(y_pcb_bot, yt + 1, full_h))

        # 1. Solid rectangular mask for the bare PCB body
        board_mask[yt:yb, xl:xr] = 255

        # 2. Segment the 3 metallic pins in the bottom extension zone below yb
        pin_zone_top = max(yt, yb - max(4, int((full_h - yb) * 0.25)))
        pin_slice = full_crop_rgb[pin_zone_top:, :]
        sh, sw = pin_slice.shape[:2]
        pin_end_margin = float(max(0, full_h - yb)) / float(max(full_h, 1))

        if sh >= 12 and sw >= 16 and full_h > yb + 4:
            gray = cv2.cvtColor(pin_slice, cv2.COLOR_RGB2GRAY).astype(np.float32)
            kw = max(7, round(sw * 0.11)) | 1
            bg_h = cv2.morphologyEx(
                gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
            )
            bg_h = cv2.GaussianBlur(bg_h, (kw, 5), 0)
            bg_ref = np.maximum(bg_h, 30.0)

            rel_dark = np.clip((bg_ref - gray) / np.maximum(bg_ref, 25.0), 0.0, 1.0)
            sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            pin_w_px = max(3, round(sw * 0.045)) | 1
            sobel_merged = cv2.GaussianBlur(sobel_x, (pin_w_px, 5), 0) / np.maximum(bg_ref, 35.0)
            pin_resp = np.clip((rel_dark * 1.55 + sobel_merged * 0.50) * 255.0, 0, 255).astype(np.uint8)

            hdr_offset = max(2, yb - pin_zone_top + 2)
            pin_resp[:hdr_offset, :] = 0
            pin_resp[:, : max(1, int(sw * 0.08))] = 0
            pin_resp[:, min(sw, int(sw * 0.92)) :] = 0

            sub = pin_resp[hdr_offset : int(sh * 0.90), int(sw * 0.10) : int(sw * 0.90)]
            if sub.size > 0:
                otsu_v, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr = int(np.clip(otsu_v * 0.88, 34, 95))
                _, pmask = cv2.threshold(pin_resp, thr, 255, cv2.THRESH_BINARY)
                pmask = cv2.morphologyEx(
                    pmask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(sw * 0.03)), max(5, int(sh * 0.08)))),
                )
                pmask = cv2.morphologyEx(
                    pmask,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(3, int(sh * 0.05)))),
                )

                num_l, labels, stats, _ = cv2.connectedComponentsWithStats(pmask, connectivity=8)
                clean_pins = np.zeros_like(pmask)
                max_y0 = hdr_offset + int((sh - hdr_offset) * 0.35)
                min_len = max(6, int((sh - hdr_offset) * 0.20))
                for lbl in range(1, num_l):
                    _, cy0, _, ch, area = stats[lbl]
                    if cy0 <= max_y0 and ch >= min_len and area >= max(12, int(sh * sw * 0.0018)):
                        clean_pins[labels == lbl] = 255

                # Bridge slight gap between yb and hdr_offset along detected pin columns
                if np.any(clean_pins > 0):
                    dilated_pins = cv2.dilate(
                        clean_pins,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
                        iterations=1,
                    )
                    board_mask[pin_zone_top:, :] = np.maximum(board_mask[pin_zone_top:, :], dilated_pins)
                    ys = np.where(clean_pins > 0)[0]
                    pin_tip_y = pin_zone_top + int(ys.max())
                    pin_end_margin = float(max(0, full_h - 1 - pin_tip_y)) / float(max(full_h, 1))

        return board_mask, round(pin_end_margin, 4)

    def resize_for_model(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize image to target_size (e.g. 256, 256)."""
        target_w, target_h = self.target_size

        if self.resize_mode == "stretch":
            return cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # "pad" letterbox mode: preserve aspect ratio, center on black background
        h, w = image_rgb.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = round(w * scale), round(h * scale)

        resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        y_offset = (target_h - nh) // 2
        x_offset = (target_w - nw) // 2
        canvas[y_offset : y_offset + nh, x_offset : x_offset + nw] = resized
        return canvas

    def resize_mask_for_model(self, mask_u8: np.ndarray) -> np.ndarray:
        """Resize binary mask to target_size (e.g. 256, 256) matching resize_for_model."""
        target_w, target_h = self.target_size
        if self.resize_mode == "stretch":
            return cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        h, w = mask_u8.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - nh) // 2
        x_offset = (target_w - nw) // 2
        canvas[y_offset : y_offset + nh, x_offset : x_offset + nw] = resized
        return canvas

    def process(self, image_rgb: np.ndarray) -> PreprocessResult:
        """Run shadow-resistant segmentation, deskew, 180-deg orientation, pin extension, and masking."""
        if image_rgb is None or image_rgb.size == 0:
            return PreprocessResult(
                success=False,
                error_message="Ảnh đầu vào rỗng hoặc không hợp lệ.",
            )

        best_cnt, _, ratio_u8, bg_rgb = self._segment_board_with_flat_field(image_rgb)
        if best_cnt is None:
            return PreprocessResult(
                success=False,
                error_message="Không tìm thấy bo mạch hợp lệ trong khung hình.",
            )

        warnings: list[str] = []
        init_rect = cv2.minAreaRect(best_cnt)
        angle = float(init_rect[2])
        corners = canonicalize_rect_corners(cv2.boxPoints(init_rect))
        w_init = max(16, round(float(np.linalg.norm(corners[0] - corners[1]))))
        h_init = max(64, round(float(np.linalg.norm(corners[0] - corners[3]))))

        # Step 1: Warp initial PCB candidate to determine canonical 180-deg upright orientation FIRST
        init_crop = self._warp_from_corners(
            image_rgb, corners, w_init, h_init, border_mode=cv2.BORDER_REPLICATE
        )
        rotation_angle = angle
        if self.is_upside_down(init_crop):
            corners = np.roll(corners, 2, axis=0)
            rotation_angle = (rotation_angle + 180.0) % 360.0

        # Step 2: Refine bare PCB bounds (pcb_rect) and sub-degree angle in upright flat-field ratio space
        init_ratio = self._warp_from_corners(
            ratio_u8, corners, w_init, h_init, border_mode=cv2.BORDER_CONSTANT, border_value=255
        )
        valid_ys: list[float] = []
        valid_xmids: list[float] = []
        for ry in range(int(h_init * 0.14), int(h_init * 0.76)):
            xs_dark = np.where(init_ratio[ry, :] < 120)[0]
            if xs_dark.size >= max(8, int(w_init * 0.55)):
                valid_ys.append(float(ry))
                valid_xmids.append(0.5 * float(xs_dark[0] + xs_dark[-1]))
        if len(valid_ys) >= 12:
            slope = float(np.polyfit(np.asarray(valid_ys), np.asarray(valid_xmids), 1)[0])
            if 0.002 < abs(slope) < 0.12:
                v_tr0 = corners[1] - corners[0]
                v_lg0 = corners[3] - corners[0]
                len_tr0 = max(float(np.linalg.norm(v_tr0)), 1e-6)
                len_lg0 = max(float(np.linalg.norm(v_lg0)), 1e-6)
                u_tr0 = v_tr0 / len_tr0
                u_lg0 = v_lg0 / len_lg0

                u_lg_new = u_lg0 + slope * u_tr0
                u_lg_new = u_lg_new / max(float(np.linalg.norm(u_lg_new)), 1e-6)
                u_tr_new = u_tr0 - float(np.dot(u_tr0, u_lg_new)) * u_lg_new
                u_tr_new = u_tr_new / max(float(np.linalg.norm(u_tr_new)), 1e-6)

                center_pt = np.mean(corners, axis=0)
                hw = 0.5 * len_tr0
                hh = 0.5 * len_lg0
                corners = np.stack(
                    [
                        center_pt - hw * u_tr_new - hh * u_lg_new,
                        center_pt + hw * u_tr_new - hh * u_lg_new,
                        center_pt + hw * u_tr_new + hh * u_lg_new,
                        center_pt - hw * u_tr_new + hh * u_lg_new,
                    ],
                    axis=0,
                ).astype(np.float32)
                rotation_angle = rotation_angle + float(np.degrees(np.arctan(slope)))
                init_ratio = self._warp_from_corners(
                    ratio_u8, corners, w_init, h_init, border_mode=cv2.BORDER_CONSTANT, border_value=255
                )

        mid_band = init_ratio[int(h_init * 0.12) : int(h_init * 0.72), :]
        col_p30 = np.percentile(mid_band, 30, axis=0)
        dark_cols = np.where(col_p30 < 120)[0]
        x0_frac = float(dark_cols[0]) / w_init if dark_cols.size > int(w_init * 0.5) else 0.0
        x1_frac = float(dark_cols[-1] + 1) / w_init if dark_cols.size > int(w_init * 0.5) else 1.0

        pw = max(8, int((x1_frac - x0_frac) * w_init))
        px0 = int(x0_frac * w_init)
        px1 = int(x1_frac * w_init)
        sh_l = init_ratio[:, px0 + max(1, int(pw * 0.04)) : px0 + max(2, int(pw * 0.22))]
        sh_r = init_ratio[:, px1 - max(2, int(pw * 0.22)) : px1 - max(1, int(pw * 0.04))]
        shoulder_val = 0.5 * (np.median(sh_l, axis=1) + np.median(sh_r, axis=1))

        y_search_start = int(h_init * 0.72)
        dark_shoulder_rows = np.where(shoulder_val[y_search_start:] < 130)[0]
        if dark_shoulder_rows.size > 0:
            y_pcb_bot_frac = float(y_search_start + dark_shoulder_rows[-1] + 1) / h_init
        else:
            y_pcb_bot_frac = 1.0
        y_pcb_bot_frac = float(np.clip(y_pcb_bot_frac, 0.80, 1.0))

        top_rows = np.where(shoulder_val[: int(h_init * 0.20)] < 130)[0]
        y_pcb_top_frac = float(top_rows[0]) / h_init if top_rows.size > 0 else 0.0
        y_pcb_top_frac = float(np.clip(y_pcb_top_frac, 0.0, 0.12))

        p0 = corners[0]
        v_trans = corners[1] - corners[0]
        v_long = corners[3] - corners[0]

        pcb_tl = p0 + x0_frac * v_trans + y_pcb_top_frac * v_long
        pcb_tr = p0 + x1_frac * v_trans + y_pcb_top_frac * v_long
        pcb_br = p0 + x1_frac * v_trans + y_pcb_bot_frac * v_long
        pcb_bl = p0 + x0_frac * v_trans + y_pcb_bot_frac * v_long
        pcb_corners = np.stack([pcb_tl, pcb_tr, pcb_br, pcb_bl], axis=0).astype(np.float32)

        pcb_w = max(1.0, float(np.linalg.norm(pcb_tr - pcb_tl)))
        pcb_h = max(1.0, float(np.linalg.norm(pcb_bl - pcb_tl)))
        pcb_ar = pcb_h / pcb_w
        if abs(pcb_ar - self.pcb_aspect_ratio_expected) > self.pcb_aspect_ratio_tolerance:
            warnings.append(
                f"pcb aspect ratio out of nominal range ({pcb_ar:.2f} vs {self.pcb_aspect_ratio_expected:.1f}±{self.pcb_aspect_ratio_tolerance:.2f})"
            )

        u_trans = (pcb_tr - pcb_tl) / pcb_w
        u_long = (pcb_bl - pcb_tl) / pcb_h

        # Step 3: Extend pcb_rect asymmetrically ONLY AFTER upright orientation is known:
        # - Head (top): head_margin_ratio * pcb_h
        # - Left/Right sides: side_margin_ratio * pcb_w
        # - Tail (bottom / 3 pins): pin_extension_ratio * pcb_h
        head_ext = self.head_margin_ratio * pcb_h
        side_ext = self.side_margin_ratio * pcb_w
        pin_ext = self.pin_extension_ratio * pcb_h

        full_tl = pcb_tl - side_ext * u_trans - head_ext * u_long
        full_tr = pcb_tr + side_ext * u_trans - head_ext * u_long
        full_br = pcb_br + side_ext * u_trans + pin_ext * u_long
        full_bl = pcb_bl - side_ext * u_trans + pin_ext * u_long
        full_corners = np.stack([full_tl, full_tr, full_br, full_bl], axis=0).astype(np.float32)

        full_w = max(16, round(pcb_w + 2.0 * side_ext))
        full_h = max(64, round(pcb_h + head_ext + pin_ext))

        img_h, img_w = image_rgb.shape[:2]
        if (
            np.any(full_corners[:, 0] < 0)
            or np.any(full_corners[:, 0] >= img_w)
            or np.any(full_corners[:, 1] < 0)
            or np.any(full_corners[:, 1] >= img_h)
        ):
            warnings.append("board near frame edge")

        # Warp using estimated paper background color for any out-of-frame extension so pins are never clipped
        aligned = self._warp_from_corners(
            image_rgb,
            full_corners,
            full_w,
            full_h,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=bg_rgb,
        )

        # Step 4: Construct board_mask (PCB + 3 pins) in crop coordinates & measure pin_end_margin
        x_pcb_left = round(side_ext)
        x_pcb_right = full_w - round(side_ext)
        y_pcb_top = round(head_ext)
        y_pcb_bot = round(head_ext + pcb_h)
        board_mask, pin_end_margin = self._build_crop_board_mask_and_pin_margin(
            aligned, x_pcb_left, x_pcb_right, y_pcb_top, y_pcb_bot
        )

        # Step 5: Illumination normalization (Gray-World White Balance + CLAHE)
        normalized = aligned.copy()
        use_clahe = bool(self.apply_clahe and self.apply_clahe_flag)
        if use_clahe or self.enable_white_balance:
            normalized = normalize_illumination(
                normalized,
                enable_white_balance=self.enable_white_balance,
                enable_clahe=use_clahe,
                clahe_clip_limit=self.clahe_clip_limit,
            )

        # Aspect-preserved fixed-size full_board_rgb (e.g. 640x144x3) for PinInspector & high-res review
        fb_w, fb_h = int(self.full_board_size[0]), int(self.full_board_size[1])
        full_board_rgb = cv2.resize(normalized, (fb_w, fb_h), interpolation=cv2.INTER_AREA)
        full_board_mask = cv2.resize(board_mask, (fb_w, fb_h), interpolation=cv2.INTER_NEAREST)

        # Build canonical masked_for_model from full_board_rgb so every board has identical (58x256) geometry
        masked_for_model = full_board_rgb.copy()
        fg = full_board_mask > 0
        if np.any(fg) and (use_clahe or self.enable_white_balance):
            lab = cv2.cvtColor(masked_for_model, cv2.COLOR_RGB2LAB).astype(np.float32)
            l_vals = lab[fg, 0]
            mu_l = float(np.mean(l_vals))
            std_l = max(float(np.std(l_vals)), 10.0)
            lab[fg, 0] = np.clip((l_vals - mu_l) * (46.0 / std_l) + 74.0, 0.0, 255.0)
            lab[fg, 1] = np.clip(lab[fg, 1] - (float(np.mean(lab[fg, 1])) - 128.0), 0.0, 255.0)
            lab[fg, 2] = np.clip(lab[fg, 2] - (float(np.mean(lab[fg, 2])) - 128.0), 0.0, 255.0)
            masked_for_model = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        if self.mask_background:
            masked_for_model[~fg] = np.asarray(self.background_fill_rgb, dtype=np.uint8)

        model_input = self.resize_for_model(masked_for_model)
        model_board_mask = self.resize_mask_for_model(full_board_mask)
        if self.mask_background:
            model_input[model_board_mask == 0] = np.asarray(self.background_fill_rgb, dtype=np.uint8)

        return PreprocessResult(
            success=True,
            cropped_rgb=aligned,
            normalized_rgb=model_input,
            bbox_box_points=full_corners.astype(np.int32),
            rotation_angle=round(rotation_angle, 2),
            error_message=None,
            full_board_rgb=full_board_rgb,
            board_mask=model_board_mask,
            full_board_mask=full_board_mask,
            pcb_bbox_points=pcb_corners.astype(np.int32),
            pin_end_margin=pin_end_margin,
            warnings=warnings,
        )


def create_board_aligner(
    config_path: Path | str | None = Path("configs/patchcore_chip.yaml"),
    **overrides: Any,
) -> BoardAligner:
    """Create a BoardAligner configured from YAML so training and inference share identical settings."""
    prep_cfg: dict[str, Any] = {}
    if config_path is not None:
        cfg_file = Path(config_path)
        if cfg_file.exists():
            try:
                with cfg_file.open("r", encoding="utf-8") as fh:
                    raw_cfg = yaml.safe_load(fh) or {}
                    prep_cfg = raw_cfg.get("preprocess", {}) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Failed to parse preprocess config %s: %s", cfg_file, exc)

    img_sz = prep_cfg.get("image_size", [256, 256])
    fb_sz = prep_cfg.get("full_board_size", [144, 640])
    bg_fill = prep_cfg.get("background_fill_rgb", [0, 0, 0])
    margin_default = float(prep_cfg.get("crop_margin_ratio", 0.025))
    params: dict[str, Any] = {
        "target_size": (int(img_sz[0]), int(img_sz[1])),
        "full_board_size": (int(fb_sz[0]), int(fb_sz[1])),
        "apply_clahe": bool(prep_cfg.get("apply_clahe", True)),
        "clahe_clip_limit": float(prep_cfg.get("clahe_clip_limit", 1.5)),
        "crop_margin_ratio": margin_default,
        "head_margin_ratio": float(prep_cfg.get("head_margin_ratio", margin_default)),
        "side_margin_ratio": float(prep_cfg.get("side_margin_ratio", 0.03)),
        "pin_extension_ratio": float(prep_cfg.get("pin_extension_ratio", 0.22)),
        "pcb_aspect_ratio_expected": float(prep_cfg.get("pcb_aspect_ratio_expected", 4.0)),
        "pcb_aspect_ratio_tolerance": float(prep_cfg.get("pcb_aspect_ratio_tolerance", 0.65)),
        "mask_background": bool(prep_cfg.get("mask_background", True)),
        "background_fill_rgb": (int(bg_fill[0]), int(bg_fill[1]), int(bg_fill[2])),
        "min_area_ratio": float(prep_cfg.get("min_contour_area_ratio", 0.008)),
        "max_area_ratio": float(prep_cfg.get("max_contour_area_ratio", 0.70)),
        "min_aspect_ratio": float(prep_cfg.get("min_aspect_ratio", 2.0)),
        "max_aspect_ratio": float(prep_cfg.get("max_aspect_ratio", 6.0)),
        "resize_mode": str(prep_cfg.get("resize_mode", "pad")),
        "enable_white_balance": bool(prep_cfg.get("enable_white_balance", True)),
    }
    params.update(overrides)
    return BoardAligner(**params)


def preprocess_board_image(
    image_rgb: np.ndarray,
    aligner: Any | None = None,
    config_path: Path | str | None = Path("configs/patchcore_chip.yaml"),
) -> PreprocessResult:
    """Single shared preprocessing function enforced across both training and inference."""
    active_aligner = aligner if aligner is not None else create_board_aligner(config_path=config_path)
    return active_aligner.process(image_rgb)


def main() -> None:
    """CLI interface for running board alignment."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Board alignment and deskew tool")
    parser.add_argument("--input", required=True, type=str, help="Path to input image")
    parser.add_argument("--output", required=True, type=str, help="Path to output image")
    parser.add_argument(
        "--resize-mode",
        choices=["pad", "stretch"],
        default="pad",
        help="Resize mode for normalized model input",
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=1.5,
        help="CLAHE clip limit (default: 1.5)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return

    bgr = cv2.imread(str(input_path))
    if bgr is None:
        logger.error("Cannot read image at %s", input_path)
        return

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    aligner = create_board_aligner(
        clahe_clip_limit=args.clahe_clip_limit,
        resize_mode=args.resize_mode,
    )

    result = preprocess_board_image(rgb, aligner=aligner)
    if not result.success or result.normalized_rgb is None:
        logger.error("Alignment Failed: %s", result.error_message)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_bgr = cv2.cvtColor(result.normalized_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out_bgr)
    logger.info("Successfully processed %s -> saved normalized board to %s", input_path, output_path)
    logger.info(
        "Rotation angle: %s deg | Cropped shape: %s | pin_end_margin: %.3f | warnings: %s",
        result.rotation_angle,
        result.cropped_rgb.shape if result.cropped_rgb is not None else None,
        result.pin_end_margin,
        result.warnings,
    )


if __name__ == "__main__":
    main()
