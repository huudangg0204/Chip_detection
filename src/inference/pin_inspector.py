"""Dedicated 3-pin geometric and occlusion inspector for LM393 sensor boards (Problem 3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.schemas import PinCheckResult

logger = logging.getLogger(__name__)

PIN_SLOT_NAMES: tuple[str, str, str] = ("G (Trái)", "V+ (Giữa)", "S (Phải)")
DEFAULT_SLOT_CENTERS_X: tuple[float, float, float] = (58.0, 116.0, 174.0)

DEFAULT_PIN_THRESHOLDS: dict[str, Any] = {
    "expected_pin_count": 3,
    "roi_y_start_ratio": 0.78,
    "roi_size": [256, 256],
    "min_length_px": 76.0,
    "max_length_px": 175.0,
    "mean_length_px": 127.6,
    "min_width_px": 8.0,
    "max_width_px": 45.0,
    "max_angle_deg": 12.5,
    "mean_angle_deg": 3.2,
    "min_spacing_px": 42.0,
    "max_spacing_px": 70.0,
    "mean_spacing_px": 55.6,
    "max_spacing_asymmetry_px": 20.0,
    "max_dark_occlusion_width_px": 34.0,
    "slot_centers_x": list(DEFAULT_SLOT_CENTERS_X),
}


class PinInspector:
    """Layer-1 deterministic geometric & occlusion inspector for the 3 bottom header pins (G, V+, S)."""

    def __init__(
        self,
        metadata_path: Path | str = Path("weights/pin_metadata.json"),
        roi_y_start_ratio: float = 0.78,
        roi_size: tuple[int, int] = (256, 256),
        default_sensitivity: float = 1.0,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.roi_y_start_ratio = float(roi_y_start_ratio)
        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        self.default_sensitivity = float(default_sensitivity)
        self.thresholds: dict[str, Any] = dict(DEFAULT_PIN_THRESHOLDS)
        self.load_metadata()

    def load_metadata(self) -> dict[str, Any]:
        """Load calibrated pin thresholds from weights/pin_metadata.json if present."""
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self.thresholds.update(data)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to load pin metadata %s: %s", self.metadata_path, exc)
        return self.thresholds

    def extract_pin_roi(self, full_board_rgb: np.ndarray) -> np.ndarray:
        """Crop the bottom header + pin zone from full_board_rgb and resize to roi_size (256x256)."""
        h, _w = full_board_rgb.shape[:2]
        y0 = int(np.clip(round(h * self.roi_y_start_ratio), 0, max(0, h - 8)))
        roi_crop = full_board_rgb[y0:, :]
        rw, rh = self.roi_size
        return cv2.resize(roi_crop, (rw, rh), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def segment_and_measure_pins(pin_roi_rgb: np.ndarray) -> dict[str, Any]:
        """Segment metallic pins in pin_roi_rgb (256x256x3) while rejecting pin & board shadows."""
        hp, wp = pin_roi_rgb.shape[:2]
        gray = cv2.cvtColor(pin_roi_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 1. Locate bottom edge of the black PCB header in top 42% of pin_roi
        paper_ref = float(np.percentile(gray[int(hp * 0.85) :, :], 75))
        top_band = gray[: int(hp * 0.42), int(wp * 0.12) : int(wp * 0.88)]
        dark_frac = np.mean(top_band < (paper_ref * 0.42), axis=1)
        pcb_rows = np.where(dark_frac > 0.55)[0]
        if pcb_rows.size > 0:
            header_bot = int(pcb_rows[-1]) + 5
        else:
            header_bot = int(hp * 0.22)
        header_bot = int(np.clip(header_bot, int(hp * 0.15), int(hp * 0.25)))

        # 2. Local horizontal background closing (kw ~ 10% of wp) subtracts smooth shadows
        kw = max(11, round(wp * 0.10)) | 1
        ker_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
        bg_h = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, ker_h)
        bg_h = cv2.GaussianBlur(bg_h, (kw, 5), 0)
        bg_ref = np.maximum(bg_h, 30.0)

        rel_dark = np.clip((bg_ref - gray) / np.maximum(bg_ref, 25.0), 0.0, 1.0)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        pin_w_px = max(5, round(wp * 0.045)) | 1
        sobel_merged = cv2.GaussianBlur(sobel_x, (pin_w_px, 5), 0) / np.maximum(bg_ref, 35.0)

        pin_resp = np.clip((rel_dark * 1.55 + sobel_merged * 0.50) * 255.0, 0, 255).astype(np.uint8)

        active = pin_resp.copy()
        active[:header_bot, :] = 0

        sub = active[header_bot : int(hp * 0.88), int(wp * 0.10) : int(wp * 0.90)]
        otsu_v, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(np.clip(otsu_v * 0.88, 34, 95))

        _, mask = cv2.threshold(active, thr, 255, cv2.THRESH_BINARY)

        central_fill = np.mean(mask[:, int(wp * 0.12) : int(wp * 0.88)] > 0, axis=1)
        for r in range(header_bot, min(hp, header_bot + 18)):
            if central_fill[r] > 0.42:
                mask[r, :] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))

        # 3. Check for solid dark foreign tape / solder bridge occlusion across pins (e.g. anomaly_test_003)
        kw_wide = max(21, round(wp * 0.35)) | 1
        bg_wide = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw_wide, 3))
        )
        bg_wide = cv2.GaussianBlur(bg_wide, (kw_wide, 5), 0)
        very_dark = ((bg_wide - gray) / np.maximum(bg_wide, 30.0)) > 0.52
        very_dark[: header_bot + 4, :] = False
        very_dark[int(hp * 0.85) :, :] = False
        mid_dark = (
            very_dark[header_bot + 10 : int(hp * 0.75), int(wp * 0.12) : int(wp * 0.88)].astype(np.uint8)
            * 255
        )
        mid_dark = cv2.morphologyEx(
            mid_dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        n_d, _, stats_d, centroids_d = cv2.connectedComponentsWithStats(mid_dark, connectivity=8)
        max_dark_blob_w = 0
        dark_blob_x = None
        for i in range(1, n_d):
            if stats_d[i][3] > 25 and int(stats_d[i][2]) > max_dark_blob_w:
                max_dark_blob_w = int(stats_d[i][2])
                dark_blob_x = float(centroids_d[i][0] + int(wp * 0.12))

        # 4. Extract elongated pin connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_start_y = header_bot + int((hp - header_bot) * 0.32)
        min_area = max(75, int(hp * wp * 0.0018))

        raw_comps: list[dict[str, Any]] = []
        for lbl in range(1, num_labels):
            x, y, cw, ch, area = stats[lbl]
            if y <= max_start_y and area >= min_area:
                pts = np.column_stack(np.where(labels == lbl))
                pts_xy = pts[:, ::-1].astype(np.float32)
                vx, vy, x0, y0 = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
                vx_f, vy_f = float(vx[0]), float(vy[0])
                if vy_f < 0:
                    vx_f, vy_f = -vx_f, -vy_f
                angle_deg = float(np.degrees(np.arctan2(abs(vx_f), max(vy_f, 1e-6))))
                rect = cv2.minAreaRect(pts_xy)
                rw, rh = rect[1]
                length_px = float(max(rw, rh))
                width_px = float(min(rw, rh))
                if length_px < max(16.0, (hp - header_bot) * 0.22):
                    continue

                top_y_cut = y + max(4, int(ch * 0.25))
                root_xs = pts_xy[pts_xy[:, 1] <= top_y_cut, 0]
                root_x = float(np.mean(root_xs)) if root_xs.size > 0 else float(centroids[lbl][0])

                if not (wp * 0.10 <= root_x <= wp * 0.90):
                    continue

                strength = float(np.mean(pin_resp[labels == lbl])) * float(length_px)
                raw_comps.append(
                    {
                        "label": lbl,
                        "x_center": round(float(centroids[lbl][0]), 1),
                        "root_x": round(root_x, 1),
                        "y_top": int(y),
                        "y_bot": int(y + ch),
                        "length_px": round(length_px, 1),
                        "height_px": round(float(ch), 1),
                        "width_px": round(width_px, 1),
                        "angle_deg": round(angle_deg, 1),
                        "area": int(area),
                        "strength": strength,
                        "bbox": (int(x), int(y), int(cw), int(ch)),
                        "line_vec": (vx_f, vy_f, float(x0[0]), float(y0[0])),
                    }
                )

        # Sort left-to-right and suppress parallel shadow duplicates within < 13% of wp
        raw_comps.sort(key=lambda c: c["root_x"])
        comps: list[dict[str, Any]] = []
        for c in raw_comps:
            if comps and abs(c["root_x"] - comps[-1]["root_x"]) < wp * 0.13:
                if c["strength"] > comps[-1]["strength"]:
                    comps[-1] = c
            else:
                comps.append(c)

        if len(comps) > 3:
            comps = sorted(comps, key=lambda c: c["strength"], reverse=True)[:3]
            comps.sort(key=lambda c: c["root_x"])

        clean_mask = np.zeros_like(mask)
        for c in comps:
            clean_mask[labels == c["label"]] = 255

        return {
            "comps": comps,
            "mask": clean_mask,
            "header_bot": header_bot,
            "max_dark_blob_w": max_dark_blob_w,
            "dark_blob_x": dark_blob_x,
        }

    def inspect(
        self,
        full_board_rgb: np.ndarray,
        board_mask: np.ndarray | None = None,
        sensitivity: float | None = None,
    ) -> PinCheckResult:
        """Run complete 3-pin inspection on full_board_rgb and return PinCheckResult."""
        del board_mask  # Optional mask input per spec signature
        if full_board_rgb is None or full_board_rgb.size == 0:
            return PinCheckResult(
                pins_ok=False,
                pin_count=0,
                per_pin=[],
                message="Ảnh full_board_rgb rỗng khi kiểm tra 3 chân.",
            )

        sens = float(sensitivity) if sensitivity is not None else self.default_sensitivity
        sens = float(np.clip(sens, 0.4, 2.5))
        # Higher sensitivity -> tighter tolerances (tol_scale < 1.0); lower sensitivity -> looser tolerances
        tol_scale = 1.0 / sens

        pin_roi = self.extract_pin_roi(full_board_rgb)
        seg = self.segment_and_measure_pins(pin_roi)
        comps = seg["comps"]
        header_bot = int(seg["header_bot"])
        max_dark_w = int(seg["max_dark_blob_w"])
        dark_blob_x = seg["dark_blob_x"]

        mean_len = float(self.thresholds.get("mean_length_px", 127.6))
        base_min_len = float(self.thresholds.get("min_length_px", 76.0))
        min_len_thr = max(35.0, mean_len - (mean_len - base_min_len) * tol_scale)
        max_angle_thr = float(self.thresholds.get("max_angle_deg", 12.5)) * tol_scale
        mean_spacing = float(self.thresholds.get("mean_spacing_px", 55.6))
        spacing_tol = max(8.0, (mean_spacing - float(self.thresholds.get("min_spacing_px", 42.0))) * tol_scale)
        min_spacing_thr = mean_spacing - spacing_tol
        max_spacing_thr = mean_spacing + spacing_tol
        max_dark_w_thr = float(self.thresholds.get("max_dark_occlusion_width_px", 34.0)) * max(0.85, tol_scale)

        slot_centers = list(self.thresholds.get("slot_centers_x", DEFAULT_SLOT_CENTERS_X))
        if len(comps) == 3:
            slot_assignment: list[dict[str, Any] | None] = [comps[0], comps[1], comps[2]]
        else:
            # Assign detected components to closest of the 3 canonical slots (G, V+, S)
            slot_assignment = [None, None, None]
            used_slots: set[int] = set()
            for comp in comps:
                dists = [
                    (abs(comp["root_x"] - slot_centers[s]), s)
                    for s in range(3)
                    if s not in used_slots
                ]
                if dists:
                    _, best_s = min(dists, key=lambda x: x[0])
                    slot_assignment[best_s] = comp
                    used_slots.add(best_s)

        per_pin: list[dict[str, Any]] = []
        fail_summaries: list[str] = []
        overlay = pin_roi.copy()
        cv2.line(overlay, (0, header_bot), (overlay.shape[1] - 1, header_bot), (80, 140, 255), 1)

        for idx in range(3):
            slot_name = PIN_SLOT_NAMES[idx]
            comp = slot_assignment[idx]
            reasons: list[str] = []

            if comp is None:
                reasons.append("thiếu chân hoặc gãy/cong gập")
                per_pin.append(
                    {
                        "pin_index": idx + 1,
                        "pin_name": slot_name,
                        "status": "FAIL",
                        "ok": False,
                        "length_px": 0.0,
                        "width_px": 0.0,
                        "angle_deg": 0.0,
                        "x_center": round(float(slot_centers[idx]), 1),
                        "spacing_px": None,
                        "reasons": reasons,
                        "reason_str": ", ".join(reasons),
                    }
                )
                fail_summaries.append(f"Chân {idx + 1} ({slot_name}): {', '.join(reasons)}")
                cx = round(slot_centers[idx])
                cv2.rectangle(overlay, (max(4, cx - 16), header_bot + 4), (min(251, cx + 16), header_bot + 110), (235, 40, 40), 2)
                cv2.putText(
                    overlay,
                    f"P{idx + 1}: THIEU",
                    (max(2, cx - 26), header_bot + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 50, 50),
                    1,
                    cv2.LINE_AA,
                )
                continue

            length_px = float(comp["length_px"])
            width_px = float(comp["width_px"])
            angle_deg = float(comp["angle_deg"])
            root_x = float(comp["root_x"])

            if length_px < min_len_thr:
                reasons.append(f"ngắn ({length_px:.0f}px < {min_len_thr:.0f}px)")
            if angle_deg > max_angle_thr:
                reasons.append(f"cong ({angle_deg:.1f}° > {max_angle_thr:.1f}°)")

            # Check spacing to previous pin if both exist
            spacing_val: float | None = None
            if idx > 0 and slot_assignment[idx - 1] is not None:
                prev_x = float(slot_assignment[idx - 1]["root_x"])
                spacing_val = round(root_x - prev_x, 1)
                if spacing_val < min_spacing_thr or spacing_val > max_spacing_thr:
                    reasons.append(f"lệch khoảng cách ({spacing_val:.1f}px)")

            # Check foreign dark tape / bridge occlusion near this pin slot
            if (
                max_dark_w > max_dark_w_thr
                and dark_blob_x is not None
                and (abs(dark_blob_x - root_x) < 42.0 or idx == 1)
            ):
                reasons.append(f"dị vật/dính chân (rộng {max_dark_w}px)")

            pin_ok = len(reasons) == 0
            status_str = "OK" if pin_ok else "FAIL"
            if not pin_ok:
                fail_summaries.append(f"Chân {idx + 1} ({slot_name}): {', '.join(reasons)}")

            per_pin.append(
                {
                    "pin_index": idx + 1,
                    "pin_name": slot_name,
                    "status": status_str,
                    "ok": pin_ok,
                    "length_px": length_px,
                    "width_px": width_px,
                    "angle_deg": angle_deg,
                    "x_center": float(comp["x_center"]),
                    "root_x": root_x,
                    "spacing_px": spacing_val,
                    "reasons": reasons,
                    "reason_str": "Bình thường" if pin_ok else ", ".join(reasons),
                }
            )

            x, y, cw, ch = comp["bbox"]
            color = (34, 197, 94) if pin_ok else (239, 68, 68)
            cv2.rectangle(overlay, (x, y), (x + cw, y + ch), color, 2)
            label_txt = f"P{idx + 1}:{angle_deg:.0f}d" if pin_ok else f"P{idx + 1}:LOI"
            cv2.putText(
                overlay,
                label_txt,
                (max(2, x - 4), max(14, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        pins_ok = (len(comps) == 3) and (len(fail_summaries) == 0)
        if pins_ok:
            msg = "3/3 chân cắm đạt chuẩn (G, V+, S thẳng & đủ chiều dài)."
        else:
            msg = "Lỗi chân cắm: " + "; ".join(fail_summaries)

        return PinCheckResult(
            pins_ok=pins_ok,
            pin_count=len(comps),
            per_pin=per_pin,
            message=msg,
            pin_roi_rgb=pin_roi,
            overlay_rgb=overlay,
        )


def calibrate_pin_inspector(
    normal_full_boards: list[np.ndarray],
    output_path: Path | str = Path("weights/pin_metadata.json"),
    roi_y_start_ratio: float = 0.78,
    roi_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Learn normal 3-pin geometric statistics from Normal training boards and save to weights/pin_metadata.json."""
    inspector = PinInspector(
        metadata_path=output_path,
        roi_y_start_ratio=roi_y_start_ratio,
        roi_size=roi_size,
    )
    lengths: list[float] = []
    widths: list[float] = []
    angles: list[float] = []
    spacings: list[float] = []
    dark_ws: list[int] = []
    slot_xs: list[list[float]] = [[], [], []]

    for fb_rgb in normal_full_boards:
        if fb_rgb is None or fb_rgb.size == 0:
            continue
        pin_roi = inspector.extract_pin_roi(fb_rgb)
        seg = inspector.segment_and_measure_pins(pin_roi)
        comps = seg["comps"]
        dark_ws.append(int(seg["max_dark_blob_w"]))
        if len(comps) == 3:
            for idx, c in enumerate(comps):
                lengths.append(float(c["length_px"]))
                widths.append(float(c["width_px"]))
                angles.append(float(c["angle_deg"]))
                slot_xs[idx].append(float(c["root_x"]))
            spacings.append(float(comps[1]["root_x"] - comps[0]["root_x"]))
            spacings.append(float(comps[2]["root_x"] - comps[1]["root_x"]))

    if not lengths:
        metadata = dict(DEFAULT_PIN_THRESHOLDS)
    else:
        len_arr = np.asarray(lengths, dtype=np.float64)
        wid_arr = np.asarray(widths, dtype=np.float64)
        ang_arr = np.asarray(angles, dtype=np.float64)
        spc_arr = np.asarray(spacings, dtype=np.float64)
        drk_arr = np.asarray(dark_ws, dtype=np.float64) if dark_ws else np.asarray([20.0])

        metadata = {
            "expected_pin_count": 3,
            "roi_y_start_ratio": roi_y_start_ratio,
            "roi_size": [int(roi_size[0]), int(roi_size[1])],
            # Min/Max expanded by 20% margin as specified in docs/update.md Section 3.4
            "min_length_px": round(float(min(np.min(len_arr) * 0.84, np.mean(len_arr) - 3.0 * np.std(len_arr))), 2),
            "max_length_px": round(float(np.max(len_arr) * 1.20), 2),
            "mean_length_px": round(float(np.mean(len_arr)), 2),
            "min_width_px": round(float(np.min(wid_arr) * 0.70), 2),
            "max_width_px": round(float(np.max(wid_arr) * 1.25), 2),
            "max_angle_deg": round(float(max(12.0, np.max(ang_arr) * 1.25, np.mean(ang_arr) + 3.5 * np.std(ang_arr))), 2),
            "mean_angle_deg": round(float(np.mean(ang_arr)), 2),
            "min_spacing_px": round(float(np.min(spc_arr) * 0.86), 2),
            "max_spacing_px": round(float(np.max(spc_arr) * 1.15), 2),
            "mean_spacing_px": round(float(np.mean(spc_arr)), 2),
            "max_spacing_asymmetry_px": 20.0,
            "max_dark_occlusion_width_px": round(float(max(32.0, np.max(drk_arr) * 1.40)), 2),
            "slot_centers_x": [
                round(float(np.mean(slot_xs[i])), 1) if slot_xs[i] else DEFAULT_SLOT_CENTERS_X[i]
                for i in range(3)
            ],
            "num_calibrated_boards": len(normal_full_boards),
        }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Saved calibrated pin metadata to %s", out_file)
    return metadata
