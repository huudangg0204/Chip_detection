"""PatchCore Inference Wrapper and Heatmap Generator (Dev 2)."""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from src.inference.pin_inspector import PinInspector
from src.preprocessing.board_aligner import create_board_aligner, preprocess_board_image
from src.schemas import InspectionResult, PinCheckResult, PreprocessResult

logger = logging.getLogger(__name__)


def build_heatmap_overlay(
    base_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
    norm_min: float | None = None,
    norm_max: float | None = None,
) -> np.ndarray:
    """Blend a 2D float anomaly map onto an RGB board image using the JET colormap.

    Args:
        base_rgb: Cropped or normalized RGB board image of shape (H, W, 3), uint8.
        anomaly_map: 2D float array of shape (H_m, W_m).
        alpha: Blend weight for the heatmap overlay in [0.0, 1.0].
        norm_min: Optional minimum value for scaling anomaly_map to [0, 1].
        norm_max: Optional maximum value for scaling anomaly_map to [0, 1].

    Returns:
        Blended RGB uint8 image with the same height and width as base_rgb.
    """
    target_h, target_w = base_rgb.shape[:2]
    resized_map = cv2.resize(
        anomaly_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    lo = float(np.min(resized_map)) if norm_min is None else float(norm_min)
    hi = float(np.max(resized_map)) if norm_max is None else float(norm_max)
    if hi - lo > 1e-8:
        scaled = np.clip((resized_map - lo) / (hi - lo), 0.0, 1.0)
    else:
        scaled = np.zeros_like(resized_map, dtype=np.float32)

    heatmap_u8 = (scaled * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        base_rgb.astype(np.uint8),
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0.0,
    )
    # Keep letterbox padding dark if base_rgb is letterboxed
    letterbox_mask = np.all(base_rgb <= 2, axis=-1)
    if np.any(letterbox_mask) and not np.all(letterbox_mask):
        blended[letterbox_mask] = base_rgb[letterbox_mask]
    return blended


def build_board_patch_mask(
    normalized_rgb: np.ndarray,
    grid_size: tuple[int, int] = (32, 32),
    board_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build a (H_p, W_p) float32 spatial weight mask in [0.0, 1.0] for the PCB.

    Suppresses black letterbox padding and 1-patch top/bottom cut edges while keeping
    full sensitivity across the active component body (LEDs, 103 potentiometer, SMD
    resistors/capacitors, LM393 IC) and moderate sensitivity on bottom header pins.
    Returns None for non-letterboxed images (e.g. synthetic test inputs).
    """
    if normalized_rgb.ndim != 3 or normalized_rgb.shape[2] != 3:
        return None

    if board_mask is not None and board_mask.shape[:2] == normalized_rgb.shape[:2]:
        non_black = (board_mask > 0).astype(np.float32)
    else:
        letterbox = np.all(normalized_rgb <= 3, axis=-1)
        letterbox_ratio = float(np.mean(letterbox))
        if letterbox_ratio < 0.25 or letterbox_ratio > 0.95:
            return None
        non_black = (~letterbox).astype(np.float32)

    gh, gw = grid_size
    # Erode board_mask slightly (Section 2D insurance layer) so borders/shadows cannot inflate score
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(non_black, kernel, iterations=1)
    patch_presence = cv2.resize(eroded, (gw, gh), interpolation=cv2.INTER_AREA)

    mask = (patch_presence > 0.65).astype(np.float32)
    # Zero out the 1-patch top and bottom cut border rims
    mask[0, :] = 0.0
    mask[-1, :] = 0.0

    # Weight zones: top mounting/LEDs (0.85), core SMD/103/LM393 zone (1.00), bottom header pins (0.45 - checked by PinInspector)
    r_core_start = max(1, round(gh * 0.20))
    r_core_end = min(gh - 2, round(gh * 0.80))
    mask[1:r_core_start, :] *= 0.85
    mask[r_core_end + 1 : gh - 1, :] *= 0.45

    if not np.any(mask > 0):
        return None
    return mask


def compute_patchcore_anomaly_map(
    patch_features: np.ndarray,
    memory_bank: np.ndarray,
    output_size: tuple[int, int] = (256, 256),
    num_neighbors: int = 9,
    gaussian_sigma: float = 4.0,
    valid_patch_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute PatchCore image-level anomaly score and pixel-level anomaly map.

    Implements Roth et al. (CVPR 2022) nearest-neighbor patch scoring with
    neighborhood soft-max re-weighting and Gaussian spatial smoothing.

    Args:
        patch_features: Spatial grid of patch descriptors (H_p, W_p, D), float32.
        memory_bank: Coreset memory bank descriptors (M, D), float32.
        output_size: Target (height, width) for the upsampled anomaly map.
        num_neighbors: Number of nearest neighbors in memory bank for score re-weighting.
        gaussian_sigma: Standard deviation for Gaussian smoothing of the anomaly map.
        valid_patch_mask: Optional (H_p, W_p) float mask in [0, 1] to restrict scoring to PCB body.

    Returns:
        Tuple of (image_anomaly_score, smoothed_anomaly_map of shape output_size).
    """
    if patch_features.ndim != 3:
        raise ValueError(f"Expected 3D patch_features (H_p, W_p, D), got {patch_features.shape}")
    if memory_bank.ndim != 2 or memory_bank.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D memory_bank (M, D), got {memory_bank.shape}")

    h_p, w_p, dim = patch_features.shape
    if memory_bank.shape[1] != dim:
        raise ValueError(
            f"Feature dimension mismatch: patch_features has D={dim}, memory_bank has D={memory_bank.shape[1]}"
        )

    flat_patches = patch_features.reshape(-1, dim).astype(np.float32)
    bank = memory_bank.astype(np.float32)

    # Efficient squared L2 distance computation: ||p - m||^2 = ||p||^2 + ||m||^2 - 2 p.m^T
    p_norm_sq = np.sum(flat_patches * flat_patches, axis=1, keepdims=True)  # (N_p, 1)
    m_norm_sq = np.sum(bank * bank, axis=1, keepdims=True).T                # (1, M)
    sq_dists = np.maximum(p_norm_sq + m_norm_sq - 2.0 * (flat_patches @ bank.T), 0.0)
    dists = np.sqrt(sq_dists)  # (N_p, M)

    # Nearest neighbor distance for each spatial patch
    min_dists = np.min(dists, axis=1)  # (N_p,)
    if valid_patch_mask is not None and valid_patch_mask.shape == (h_p, w_p):
        min_dists = min_dists * valid_patch_mask.reshape(-1).astype(np.float32)

    max_patch_idx = int(np.argmax(min_dists))
    s_star = float(min_dists[max_patch_idx])

    # Soft-max neighborhood re-weighting around the most anomalous patch p^* (Eq. 3 in PatchCore)
    k = min(max(1, num_neighbors), bank.shape[0])
    if k > 1:
        nearest_k_dists = np.partition(dists[max_patch_idx], k - 1)[:k]
        # Shift by s_star for numerical stability
        shifted = nearest_k_dists - s_star
        exp_vals = np.exp(np.clip(shifted, -50.0, 50.0))
        denom = float(np.sum(exp_vals))
        weight = 1.0 - (1.0 / denom) if denom > 1e-12 else 1.0
        image_score = float(weight * s_star)
    else:
        image_score = s_star

    patch_score_grid = min_dists.reshape(h_p, w_p).astype(np.float32)
    out_h, out_w = output_size
    upsampled_map = cv2.resize(
        patch_score_grid,
        (out_w, out_h),
        interpolation=cv2.INTER_LINEAR,
    )
    if gaussian_sigma > 0:
        smoothed_map = gaussian_filter(upsampled_map, sigma=gaussian_sigma).astype(np.float32)
    else:
        smoothed_map = upsampled_map.astype(np.float32)

    return image_score, smoothed_map


def update_threshold_decision(
    result: InspectionResult,
    new_threshold: float,
) -> InspectionResult:
    """Re-evaluate Boolean Normal/Anomaly decision when user drags the live threshold slider.

    Runs in O(1) without re-executing preprocessing or neural network inference.
    """
    if result.status != "OK" or result.anomaly_score is None:
        return replace(result, threshold=float(new_threshold))

    board_ok = bool(result.anomaly_score <= new_threshold)
    pins_ok = True if result.pins_ok is None else bool(result.pins_ok)
    is_normal = bool(board_ok and pins_ok)

    if is_normal:
        message = "Normal (Không phát hiện bất thường)"
    elif not board_ok and not pins_ok:
        pin_msg = result.pin_result.message if result.pin_result is not None else "Lỗi chân cắm"
        message = f"Anomaly (Lỗi bề mặt bo mạch & {pin_msg})"
    elif not pins_ok:
        pin_msg = result.pin_result.message if result.pin_result is not None else "Lỗi chân cắm"
        message = f"Anomaly ({pin_msg})"
    else:
        message = "Anomaly (Phát hiện bất thường trên bo mạch)"

    return replace(
        result,
        is_normal=is_normal,
        board_ok=board_ok,
        pins_ok=pins_ok,
        threshold=float(new_threshold),
        message=message,
    )


class PatchCoreFeatureExtractor:
    """Extracts spatial patch embeddings (H_p, W_p, D) for PatchCore.

    Uses torchvision `wide_resnet50_2` (layers `layer2` + `layer3` with 3x3 local neighborhood
    aggregation) when PyTorch is available, or falls back to a multi-scale dense spatial patch
    descriptor when running in lightweight environments without PyTorch installed.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: list[str] | None = None,
        grid_size: tuple[int, int] = (32, 32),
        force_fallback: bool = False,
        offline_backbone_path: Path | str | None = None,
    ) -> None:
        self.backbone_name = backbone_name
        self.layers = layers or ["layer2", "layer3"]
        self.grid_size = grid_size
        self.force_fallback = force_fallback
        self.offline_backbone_path = (
            Path(offline_backbone_path) if offline_backbone_path is not None else None
        )
        self.backend_type = "fallback_cv"
        self._torch_model: Any = None
        self._device: Any = None
        self._hook_outputs: dict[str, Any] = {}

        if not force_fallback:
            self._try_init_torch_backbone()

    def _resolve_offline_backbone_candidates(self) -> list[Path]:
        """Return candidate local paths for cached backbone weights (for offline Kaggle/Colab runs)."""
        filename = f"{self.backbone_name}_layer3_fp16.pt"
        repo_root = Path(__file__).resolve().parents[2]
        candidates: list[Path] = []
        if self.offline_backbone_path is not None:
            candidates.append(self.offline_backbone_path)
        candidates.append(Path("weights") / filename)
        candidates.append(repo_root / "weights" / filename)
        return candidates

    def _load_offline_torch_backbone(self, torch_mod: Any, models_mod: Any) -> Any | None:
        """Instantiate torchvision backbone without downloading and load local state_dict if present."""
        for cand in self._resolve_offline_backbone_candidates():
            if not cand.is_file():
                continue
            try:
                if self.backbone_name == "resnet18":
                    backbone = models_mod.resnet18(weights=None)
                else:
                    backbone = models_mod.wide_resnet50_2(weights=None)
                raw_sd = torch_mod.load(cand, map_location="cpu", weights_only=True)
                sd_fp32 = {
                    k: (v.float() if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
                    for k, v in raw_sd.items()
                }
                backbone.load_state_dict(sd_fp32, strict=False)
                logger.info("Loaded offline backbone weights from %s", cand)
                return backbone
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                logger.warning("Failed to load offline backbone from %s: %s", cand, exc)
        return None

    def _has_torch_hub_cached_weights(self, torch_mod: Any) -> bool:
        """Check whether torchvision backbone weights already exist in local Torch Hub cache."""
        try:
            ckpt_dir = Path(torch_mod.hub.get_dir()) / "checkpoints"
            if not ckpt_dir.is_dir():
                return False
            prefix = "resnet18" if self.backbone_name == "resnet18" else "wide_resnet50_2"
            return any(ckpt_dir.glob(f"{prefix}*.pth"))
        except (OSError, AttributeError, TypeError):
            return False

    def _try_init_torch_backbone(self) -> None:
        """Initialize torchvision wide_resnet50_2 feature hooks if torch is available."""
        try:
            import torch
            from torchvision import models

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            backbone: Any = None

            # Prefer local offline bundle immediately when Torch Hub cache is empty (avoids DNS hang on offline Kaggle)
            if not self._has_torch_hub_cached_weights(torch):
                backbone = self._load_offline_torch_backbone(torch, models)

            if backbone is None:
                try:
                    if self.backbone_name == "wide_resnet50_2":
                        weights = models.Wide_ResNet50_2_Weights.DEFAULT
                        backbone = models.wide_resnet50_2(weights=weights)
                    elif self.backbone_name == "resnet18":
                        weights = models.ResNet18_Weights.DEFAULT
                        backbone = models.resnet18(weights=weights)
                    else:
                        weights = models.Wide_ResNet50_2_Weights.DEFAULT
                        backbone = models.wide_resnet50_2(weights=weights)
                except (OSError, RuntimeError) as download_exc:
                    logger.info(
                        "Online torchvision weights unavailable (%s); checking offline bundle.",
                        download_exc,
                    )
                    backbone = self._load_offline_torch_backbone(torch, models)
                    if backbone is None:
                        raise

            backbone.eval()
            backbone.to(self._device)

            def _make_hook(layer_name: str):
                def _hook(_module: Any, _inp: Any, out: Any) -> None:
                    self._hook_outputs[layer_name] = out.detach()
                return _hook

            for layer_name in self.layers:
                layer_mod = getattr(backbone, layer_name)
                layer_mod.register_forward_hook(_make_hook(layer_name))

            self._torch_model = backbone
            self.backend_type = "torch_backbone"
            logger.info("Initialized PyTorch PatchCore backbone '%s' on %s", self.backbone_name, self._device)
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            logger.debug("PyTorch backbone unavailable (%s); using dense CV patch extractor.", exc)
            self._torch_model = None
            self.backend_type = "fallback_cv"

    def extract(self, normalized_rgb: np.ndarray) -> np.ndarray:
        """Extract spatial patch descriptors of shape (H_p, W_p, D) from a (256, 256, 3) RGB image."""
        if self.backend_type == "torch_backbone" and self._torch_model is not None:
            return self._extract_torch(normalized_rgb)
        return self._extract_dense_cv(normalized_rgb)

    def _extract_torch(self, normalized_rgb: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self._hook_outputs.clear()
        img_f = normalized_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_f - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            _ = self._torch_model(tensor)
            pooled_maps = []
            target_h, target_w = self.grid_size
            for idx, layer_name in enumerate(self.layers):
                feat = self._hook_outputs[layer_name]
                # Local 3x3 neighborhood aggregation (PatchCore Section 3.1)
                feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
                if idx == 0:
                    target_h, target_w = feat.shape[2], feat.shape[3]
                elif feat.shape[2:] != (target_h, target_w):
                    feat = F.interpolate(
                        feat,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                pooled_maps.append(feat)

            concat_feat = torch.cat(pooled_maps, dim=1)  # (1, C_total, H_p, W_p)
            concat_feat = F.normalize(concat_feat, p=2, dim=1)
            patch_grid = concat_feat.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            return patch_grid

    def _extract_dense_cv(self, normalized_rgb: np.ndarray) -> np.ndarray:
        """Extract multi-scale color, gradient, and local texture patch descriptors (H_p, W_p, D)."""
        img_256 = cv2.resize(normalized_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img_256, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img_256, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        rgb = img_256.astype(np.float32) / 255.0
        gray = cv2.cvtColor(img_256, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.sqrt(gx * gx + gy * gy)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

        # Local variance & multi-scale smoothed channels to capture tiny SMD components
        blur_small = cv2.GaussianBlur(rgb, (5, 5), 1.0)
        blur_med = cv2.GaussianBlur(rgb, (11, 11), 2.5)
        high_freq = np.abs(rgb - blur_small)
        band_pass = np.abs(blur_small - blur_med)

        # Stack rich pixel channels: RGB(3) + LAB(3) + HSV(3) + HighFreq(3) + BandPass(3) + Grad(3) = 18 channels
        stack = np.concatenate(
            [
                rgb,
                lab,
                hsv,
                high_freq * 2.5,
                band_pass * 2.5,
                gx[..., None],
                gy[..., None],
                gmag[..., None],
                lap[..., None],
            ],
            axis=-1,
        ).astype(np.float32)  # (256, 256, 19)

        gh, gw = self.grid_size
        # Pool mean and std over overlapping 3x3 patch neighborhoods at (gh, gw) resolution
        mean_map = cv2.resize(
            cv2.GaussianBlur(stack, (9, 9), 2.0),
            (gw, gh),
            interpolation=cv2.INTER_AREA,
        )
        sq_map = cv2.resize(
            cv2.GaussianBlur(stack * stack, (9, 9), 2.0),
            (gw, gh),
            interpolation=cv2.INTER_AREA,
        )
        std_map = np.sqrt(np.maximum(sq_map - mean_map * mean_map, 0.0))

        # Add normalized spatial coordinate embeddings so patches only match structurally valid board zones
        ys = np.linspace(-1.0, 1.0, gh, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, gw, dtype=np.float32)[None, :]
        coord_y = np.broadcast_to(ys, (gh, gw))[..., None] * 0.35
        coord_x = np.broadcast_to(xs, (gh, gw))[..., None] * 0.35

        patch_grid = np.concatenate([mean_map, std_map, coord_y, coord_x], axis=-1).astype(np.float32)
        return patch_grid


class InspectionModel:
    """Unified wrapper combining BoardAligner preprocessing and PatchCore inference."""

    def __init__(
        self,
        weights_path: Path | str = Path("weights/patchcore_best.ckpt"),
        metadata_path: Path | str = Path("weights/metadata.json"),
        config_path: Path | str = Path("configs/patchcore_chip.yaml"),
        default_threshold: float | None = None,
        aligner: Any | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)

        cfg = self._load_yaml_config()
        meta = self._load_metadata()

        if default_threshold is not None:
            self.default_threshold = float(default_threshold)
        elif "optimal_threshold" in meta:
            self.default_threshold = float(meta["optimal_threshold"])
        else:
            self.default_threshold = float(
                cfg.get("decision", {}).get("default_threshold", 0.50)
            )

        model_cfg = cfg.get("model", {})
        self.backbone_name: str = str(model_cfg.get("backbone", "wide_resnet50_2"))
        self.layers: list[str] = list(model_cfg.get("layers", ["layer2", "layer3"]))
        self.num_neighbors: int = int(model_cfg.get("num_neighbors", 9))

        prep_cfg = cfg.get("preprocess", {})
        img_sz = prep_cfg.get("image_size", [256, 256])
        self.image_size: tuple[int, int] = (int(img_sz[0]), int(img_sz[1]))

        pin_cfg = cfg.get("pin_check", {})
        self.enable_pin_check: bool = bool(pin_cfg.get("enabled", True))
        pin_roi_sz = pin_cfg.get("roi_size", [256, 256])
        self.pin_inspector = PinInspector(
            metadata_path=Path(pin_cfg.get("metadata_path", "weights/pin_metadata.json")),
            roi_y_start_ratio=float(pin_cfg.get("roi_y_start_ratio", 0.78)),
            roi_size=(int(pin_roi_sz[0]), int(pin_roi_sz[1])),
            default_sensitivity=float(pin_cfg.get("default_sensitivity", 1.0)),
        )

        self.aligner = aligner if aligner is not None else self._resolve_aligner(prep_cfg)
        self._memory_bank: np.ndarray | None = None
        self._feature_extractor: PatchCoreFeatureExtractor | None = None
        self._anomalib_inferencer: Any = None
        self._score_min: float = float(meta.get("score_min", 0.0))
        self._score_scale: float = float(meta.get("score_scale", 1.0))
        self._is_loaded: bool = False

    def _load_yaml_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("Failed to parse config %s: %s", self.config_path, exc)
        return {}

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read metadata %s: %s", self.metadata_path, exc)
        return {}

    def _resolve_aligner(self, prep_cfg: dict[str, Any]) -> Any:
        """Instantiate shared BoardAligner configured consistently with training."""
        del prep_cfg
        return create_board_aligner(config_path=self.config_path, target_size=self.image_size)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self, model_path: Path | str | None = None) -> None:
        """Load trained PatchCore checkpoint from disk.

        Supports:
        1. PatchCore Memory Bank dictionary saved via pickle/torch/npz
        2. Anomalib Lightning checkpoint (`.ckpt`) containing `state_dict['model.memory_bank']`
        """
        if model_path is not None:
            self.weights_path = Path(model_path)

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy trọng số model tại: {self.weights_path}"
            )

        payload = self._read_checkpoint_file(self.weights_path)
        if isinstance(payload, dict) and "memory_bank" in payload:
            self._memory_bank = np.asarray(payload["memory_bank"], dtype=np.float32)
            backend_type = str(payload.get("backend_type", "auto"))
            grid_sz = tuple(payload.get("grid_size", (32, 32)))
            self._score_min = float(payload.get("score_min", self._score_min))
            self._score_scale = float(payload.get("score_scale", self._score_scale))
            if "optimal_threshold" in payload:
                self.default_threshold = float(payload["optimal_threshold"])
            ckpt_backbone = str(payload.get("backbone", self.backbone_name))
            offline_candidate = self.weights_path.parent / f"{ckpt_backbone}_layer3_fp16.pt"
            self._feature_extractor = PatchCoreFeatureExtractor(
                backbone_name=ckpt_backbone,
                layers=list(payload.get("layers", self.layers)),
                grid_size=(int(grid_sz[0]), int(grid_sz[1])),
                force_fallback=(backend_type == "fallback_cv"),
                offline_backbone_path=offline_candidate if offline_candidate.exists() else None,
            )
            self._is_loaded = True
            logger.info(
                "Loaded PatchCore memory bank (%s patches, dim=%d) from %s",
                self._memory_bank.shape[0],
                self._memory_bank.shape[1],
                self.weights_path,
            )
            return

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            for key in ("model.memory_bank", "memory_bank", "model.embedding_store"):
                if key in state_dict:
                    tensor_val = state_dict[key]
                    bank_np = (
                        tensor_val.detach().cpu().numpy()
                        if hasattr(tensor_val, "detach")
                        else np.asarray(tensor_val)
                    )
                    self._memory_bank = bank_np.astype(np.float32)
                    self._feature_extractor = PatchCoreFeatureExtractor(
                        backbone_name=self.backbone_name,
                        layers=self.layers,
                    )
                    self._is_loaded = True
                    return

        raise ValueError(
            f"Checkpoint {self.weights_path} không chứa 'memory_bank' hợp lệ của PatchCore."
        )

    @staticmethod
    def _read_checkpoint_file(path: Path) -> Any:
        """Read checkpoint using pickle, numpy, or torch.load."""
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except (OSError, pickle.PickleError, EOFError) as exc:
            logger.debug("Pickle load failed for %s: %s", path, exc)

        try:
            import torch

            return torch.load(path, map_location="cpu", weights_only=False)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Không thể đọc file checkpoint {path}: {exc}") from exc

    def predict(
        self,
        image_rgb: np.ndarray,
        threshold: float | None = None,
        preprocess_result: PreprocessResult | None = None,
        pin_sensitivity: float | None = None,
    ) -> InspectionResult:
        """Execute preprocessing, PatchCore PCB anomaly detection, and 3-pin inspection."""
        start_time = time.perf_counter()
        tau = float(threshold) if threshold is not None else self.default_threshold

        if image_rgb is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return InspectionResult(
                status="ERROR",
                threshold=tau,
                processing_time_ms=round(elapsed_ms, 2),
                message="Ảnh đầu vào rỗng hoặc không hợp lệ.",
            )

        # Step 1: Preprocess via unified shared function
        prep: PreprocessResult = (
            preprocess_result
            if preprocess_result is not None
            else preprocess_board_image(image_rgb, aligner=self.aligner)
        )
        if not prep.success or prep.normalized_rgb is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return InspectionResult(
                status="NO_BOARD_DETECTED",
                is_normal=None,
                anomaly_score=None,
                threshold=tau,
                processing_time_ms=round(elapsed_ms, 2),
                message=prep.error_message or "Không phát hiện được bo mạch trong ảnh.",
                warnings=list(getattr(prep, "warnings", []) or []),
            )

        # Step 2: Run PatchCore inference on masked normalized board image
        self._current_board_mask = getattr(prep, "board_mask", None)
        try:
            score, anomaly_map, heatmap_overlay = self._run_patchcore(
                prep.normalized_rgb,
                prep.cropped_rgb,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("PatchCore inference failed")
            return InspectionResult(
                status="ERROR",
                is_normal=None,
                anomaly_score=None,
                threshold=tau,
                cropped_board_rgb=prep.cropped_rgb,
                processing_time_ms=round(elapsed_ms, 2),
                message=f"Lỗi khi chạy mô hình PatchCore: {exc}",
                warnings=list(getattr(prep, "warnings", []) or []),
            )

        # Step 3: Run dedicated 3-pin inspection on full_board_rgb when available
        pin_res: PinCheckResult | None = None
        pins_ok = True
        full_board_rgb = getattr(prep, "full_board_rgb", None)
        full_board_mask = getattr(prep, "full_board_mask", None)
        if self.enable_pin_check and full_board_rgb is not None:
            pin_res = self.pin_inspector.inspect(
                full_board_rgb=full_board_rgb,
                board_mask=full_board_mask,
                sensitivity=pin_sensitivity,
            )
            pins_ok = bool(pin_res.pins_ok)

        board_ok = bool(score <= tau)
        if not pins_ok:
            # Ensure anomaly_score reflects pin defect for unified threshold/AUROC evaluation
            score = max(float(score), min(1.0, max(tau + 0.35, 0.85)))

        is_normal = bool(board_ok and pins_ok)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if is_normal:
            msg = "Normal (Không phát hiện bất thường — Bo mạch & 3 chân đạt chuẩn)"
        elif not board_ok and not pins_ok and pin_res is not None:
            msg = f"Anomaly (Lỗi bề mặt bo mạch & {pin_res.message})"
        elif not pins_ok and pin_res is not None:
            msg = f"Anomaly ({pin_res.message})"
        else:
            msg = "Anomaly (Phát hiện bất thường trên bề mặt bo mạch)"

        return InspectionResult(
            status="OK",
            is_normal=is_normal,
            anomaly_score=round(float(score), 4),
            threshold=round(tau, 4),
            cropped_board_rgb=prep.cropped_rgb,
            anomaly_map=anomaly_map,
            heatmap_overlay_rgb=heatmap_overlay,
            processing_time_ms=round(elapsed_ms, 2),
            message=msg,
            board_ok=board_ok,
            pins_ok=pins_ok,
            pin_result=pin_res,
            warnings=list(getattr(prep, "warnings", []) or []),
        )

    def _run_patchcore(
        self,
        normalized_rgb: np.ndarray,
        cropped_rgb: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Extract features, query memory bank, and generate heatmap overlay."""
        if not self._is_loaded or self._memory_bank is None or self._feature_extractor is None:
            if self.weights_path.exists():
                self.load()
            else:
                raise RuntimeError(
                    f"Model chưa được load và không tìm thấy file trọng số tại {self.weights_path}."
                )

        assert self._feature_extractor is not None
        assert self._memory_bank is not None

        patch_features = self._feature_extractor.extract(normalized_rgb)
        valid_mask = build_board_patch_mask(
            normalized_rgb,
            grid_size=(patch_features.shape[0], patch_features.shape[1]),
            board_mask=getattr(self, "_current_board_mask", None),
        )
        raw_score, raw_map = compute_patchcore_anomaly_map(
            patch_features=patch_features,
            memory_bank=self._memory_bank,
            output_size=self.image_size,
            num_neighbors=self.num_neighbors,
            valid_patch_mask=valid_mask,
        )

        # Normalize score and anomaly map using calibration scale from training/validation
        denom = max(self._score_scale, 1e-6)
        norm_score = float(np.clip((raw_score - self._score_min) / denom, 0.0, 1.0))
        norm_map = np.clip((raw_map - self._score_min) / denom, 0.0, 1.0).astype(np.float32)
        if valid_mask is not None:
            pixel_mask = cv2.resize(
                valid_mask,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            norm_map = (norm_map * pixel_mask).astype(np.float32)

        heatmap_overlay = build_heatmap_overlay(
            base_rgb=normalized_rgb,
            anomaly_map=norm_map,
            norm_min=0.0,
            norm_max=max(float(np.max(norm_map)), self.default_threshold * 1.4, 0.35),
        )
        return norm_score, norm_map, heatmap_overlay
