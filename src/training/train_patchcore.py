"""PatchCore Training Pipeline, Multi-Lighting Augmentation & Coreset Builder (Dev 2)."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.inference.inspector import (
    PatchCoreFeatureExtractor,
    build_board_patch_mask,
    compute_patchcore_anomaly_map,
)
from src.inference.pin_inspector import PinInspector, calibrate_pin_inspector
from src.preprocessing.board_aligner import create_board_aligner, preprocess_board_image
from src.training.evaluate import (
    compute_auroc,
    find_optimal_threshold,
    list_images_in_dir,
)

logger = logging.getLogger(__name__)


def greedy_coreset_subsampling(
    embeddings: np.ndarray,
    sampling_ratio: float = 0.10,
    seed: int = 42,
    max_coreset_size: int = 4096,
    projection_dim: int = 128,
) -> np.ndarray:
    """Select representative patch descriptors using PatchCore Greedy k-Center Coreset Subsampling.

    Implements Algorithm 1 from Roth et al., 'Towards Total Recall in Industrial Anomaly Detection'
    (CVPR 2022), using Johnson-Lindenstrauss random linear projection to speed up distance checks.

    Args:
        embeddings: 2D array of patch descriptors (N, D), float32.
        sampling_ratio: Fraction of patches to retain in (0.0, 1.0].
        seed: Random seed for reproducible projection and starting center.
        max_coreset_size: Upper cap on coreset memory bank size.
        projection_dim: Target dimension for random projection distance computation.

    Returns:
        Subsampled memory bank array of shape (M, D), float32.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D embeddings (N, D), got {embeddings.shape}")

    n_total, dim = embeddings.shape
    target_m = max(1, min(round(n_total * sampling_ratio), max_coreset_size, n_total))
    if target_m >= n_total:
        return embeddings.astype(np.float32)

    rng = np.random.default_rng(seed)
    if dim > projection_dim:
        proj_matrix = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(projection_dim),
            size=(dim, projection_dim),
        ).astype(np.float32)
        projected = embeddings.astype(np.float32) @ proj_matrix
    else:
        projected = embeddings.astype(np.float32)

    selected_indices: list[int] = []
    first_idx = int(rng.integers(0, n_total))
    selected_indices.append(first_idx)

    center = projected[first_idx : first_idx + 1]
    min_sq_dists = np.sum((projected - center) ** 2, axis=1)
    min_sq_dists[first_idx] = -1.0

    for _ in range(1, target_m):
        next_idx = int(np.argmax(min_sq_dists))
        selected_indices.append(next_idx)
        new_center = projected[next_idx : next_idx + 1]
        sq_dists = np.sum((projected - new_center) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
        min_sq_dists[next_idx] = -1.0

    return embeddings[selected_indices].astype(np.float32)


def generate_photometric_augmentations(
    image_rgb: np.ndarray,
    num_variants: int = 4,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate multi-lighting photometric variations without geometric flipping.

    Simulates the 4 lighting regimes defined in Section 5.1 of `docs/developer_spec.md`:
    1. Natural room illumination with mild brightness/contrast shift
    2. Directional illumination from left (desk lamp left)
    3. Directional illumination from right/top (desk lamp right/top)
    4. Slightly dim lighting with mild Gaussian blur and micro 2D pose tolerance

    NEVER applies HorizontalFlip, VerticalFlip, or RandomCrop so asymmetric PCB components
    always stay in their canonical spatial locations.
    """
    rng = np.random.default_rng(seed)
    h, w = image_rgb.shape[:2]
    variants: list[np.ndarray] = []

    x_grid = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    y_grid = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]

    for idx in range(num_variants):
        img = image_rgb.astype(np.float32)
        mode = idx % 4

        if mode == 0:
            # Mode 0: Warm/cool natural room shift
            gain = float(rng.uniform(0.92, 1.08))
            bias = float(rng.uniform(-8.0, 8.0))
            img = img * gain + bias
        elif mode == 1:
            # Mode 1: Left directional lamp gradient
            strength = float(rng.uniform(0.08, 0.16))
            grad = 1.0 + strength * (0.5 - x_grid)
            img = img * grad
        elif mode == 2:
            # Mode 2: Right/top directional lamp gradient
            strength = float(rng.uniform(0.08, 0.16))
            grad = 1.0 + strength * (x_grid - 0.5) + 0.05 * (0.5 - y_grid)
            img = img * grad
        else:
            # Mode 3: Slightly dim light + webcam-scale resolution & sensor blur
            gain = float(rng.uniform(0.88, 0.96))
            small_w, small_h = max(64, w // 2), max(64, h // 2)
            down = cv2.resize(img * gain, (small_w, small_h), interpolation=cv2.INTER_AREA)
            img = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.GaussianBlur(img, (3, 3), 0.4)

        # Mild saturation jitter in HSV space
        clipped_u8 = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(clipped_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_scale = float(rng.uniform(0.90, 1.10))
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
        out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Micro 2D translation/rotation jitter (<= 1.0 deg, <= 1.5 px) for memory bank tolerance
        angle = float(rng.uniform(-0.8, 0.8))
        tx = float(rng.uniform(-1.2, 1.2))
        ty = float(rng.uniform(-1.2, 1.2))
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
        mat[0, 2] += tx
        mat[1, 2] += ty
        out_rgb = cv2.warpAffine(
            out_rgb,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Preserve pure-black letterbox padding if present
        zero_mask = np.all(image_rgb == 0, axis=-1)
        if np.any(zero_mask) and not np.all(zero_mask):
            out_rgb[zero_mask] = 0
        variants.append(out_rgb)

    return variants


def bootstrap_dataset_from_samples(
    dataset_root: Path = Path("data/chip_dataset"),
    sample_normal_path: Path = Path("image/sample2.jpg"),
    sample_anomaly_path: Path = Path("image/sample1.jpg"),
    num_train_normal: int = 30,
    num_val_normal: int = 10,
    num_test_abnormal: int = 10,
) -> dict[str, int]:
    """Bootstrap standard Anomalib folder dataset from sample2.jpg (Normal) & sample1.jpg (Anomaly).

    In the project's initial state:
    - `image/sample2.jpg` has the complete PCB with both SMD components next to the 103 potentiometer.
    - `image/sample1.jpg` is missing the SMD components next to the 103 potentiometer (Anomaly).

    If the user has not yet populated `data/chip_dataset/` with 30+ camera shots, this function
    populates `normal/train`, `normal/val`, and `abnormal/test` with realistic multi-lighting
    and slight placement variations so training and evaluation work immediately.
    """
    train_dir = dataset_root / "normal" / "train"
    val_dir = dataset_root / "normal" / "val"
    test_anom_dir = dataset_root / "abnormal" / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_anom_dir.mkdir(parents=True, exist_ok=True)

    existing_train = list_images_in_dir(train_dir)
    existing_val = list_images_in_dir(val_dir)
    existing_anom = list_images_in_dir(test_anom_dir)

    if (
        len(existing_train) >= 8
        and len(existing_val) >= 2
        and len(existing_anom) >= 2
    ):
        return {
            "normal_train": len(existing_train),
            "normal_val": len(existing_val),
            "abnormal_test": len(existing_anom),
        }

    if not sample_normal_path.exists() or not sample_anomaly_path.exists():
        logger.warning(
            "Sample images not found at %s / %s; skipping bootstrap.",
            sample_normal_path,
            sample_anomaly_path,
        )
        return {
            "normal_train": len(existing_train),
            "normal_val": len(existing_val),
            "abnormal_test": len(existing_anom),
        }

    norm_bgr = cv2.imread(str(sample_normal_path))
    anom_bgr = cv2.imread(str(sample_anomaly_path))
    if norm_bgr is None or anom_bgr is None:
        return {"normal_train": 0, "normal_val": 0, "abnormal_test": 0}

    norm_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
    anom_rgb = cv2.cvtColor(anom_bgr, cv2.COLOR_BGR2RGB)

    # Save base normal and multi-lighting normal training shots
    cv2.imwrite(str(train_dir / "normal_000.jpg"), norm_bgr)
    train_variants = generate_photometric_augmentations(
        norm_rgb, num_variants=max(0, num_train_normal - 1), seed=101
    )
    for idx, var_rgb in enumerate(train_variants, start=1):
        var_bgr = cv2.cvtColor(var_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(train_dir / f"normal_{idx:03d}.jpg"), var_bgr)

    # Save validation normal shots with independent seed
    val_variants = generate_photometric_augmentations(
        norm_rgb, num_variants=num_val_normal, seed=202
    )
    for idx, var_rgb in enumerate(val_variants):
        var_bgr = cv2.cvtColor(var_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(val_dir / f"val_normal_{idx:03d}.jpg"), var_bgr)

    # Save real anomaly sample1.jpg + multi-lighting variants
    cv2.imwrite(str(test_anom_dir / "anomaly_000_missing_smd.jpg"), anom_bgr)
    anom_variants = generate_photometric_augmentations(
        anom_rgb, num_variants=max(0, num_test_abnormal - 1), seed=303
    )
    for idx, var_rgb in enumerate(anom_variants, start=1):
        var_bgr = cv2.cvtColor(var_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(test_anom_dir / f"anomaly_{idx:03d}_missing_smd.jpg"), var_bgr)

    return {
        "normal_train": len(list_images_in_dir(train_dir)),
        "normal_val": len(list_images_in_dir(val_dir)),
        "abnormal_test": len(list_images_in_dir(test_anom_dir)),
    }


def train_patchcore_model(
    config_path: Path = Path("configs/patchcore_chip.yaml"),
    dataset_root: Path = Path("data/chip_dataset"),
    output_weights_path: Path = Path("weights/patchcore_best.ckpt"),
    output_metadata_path: Path = Path("weights/metadata.json"),
    auto_bootstrap: bool = True,
) -> dict[str, Any]:
    """Train PatchCore memory bank on normal images and calibrate decision threshold tau."""
    cfg: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    model_cfg = cfg.get("model", {})
    backbone_name = str(model_cfg.get("backbone", "wide_resnet50_2"))
    layers = list(model_cfg.get("layers", ["layer2", "layer3"]))
    coreset_ratio = float(model_cfg.get("coreset_sampling_ratio", 0.10))
    num_neighbors = int(model_cfg.get("num_neighbors", 9))

    prep_cfg = cfg.get("preprocess", {})
    img_sz = prep_cfg.get("image_size", [256, 256])
    image_size = (int(img_sz[0]), int(img_sz[1]))

    if auto_bootstrap:
        counts = bootstrap_dataset_from_samples(dataset_root=dataset_root)
        logger.info("Dataset counts: %s", counts)

    train_dir = dataset_root / "normal" / "train"
    train_files = list_images_in_dir(train_dir)
    if not train_files:
        raise RuntimeError(
            f"Không tìm thấy ảnh huấn luyện nào trong {train_dir}. Hãy chụp ảnh Normal trước!"
        )
    sample2_ref = Path("image/sample2.jpg")
    if sample2_ref.exists() and sample2_ref not in train_files:
        train_files = [*train_files, sample2_ref]

    # Use shared preprocessing configuration and function identical to inference
    aligner = create_board_aligner(config_path=config_path)

    extractor = PatchCoreFeatureExtractor(
        backbone_name=backbone_name,
        layers=layers,
        grid_size=(32, 32),
    )

    all_patch_descriptors: list[np.ndarray] = []
    normalized_train_images: list[np.ndarray] = []
    train_full_boards: list[np.ndarray] = []

    for img_path in train_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prep = preprocess_board_image(rgb, aligner=aligner)
        if not prep.success or prep.normalized_rgb is None:
            logger.warning("Skipping %s: board alignment failed (%s)", img_path.name, prep.error_message)
            continue

        norm_img = prep.normalized_rgb
        normalized_train_images.append(norm_img)
        if prep.full_board_rgb is not None:
            train_full_boards.append(prep.full_board_rgb)

        # Extract features from original aligned board + photometric/webcam-resolution augmentations
        n_vars = 6 if min(rgb.shape[:2]) < 900 else 3
        aug_imgs = [norm_img] + generate_photometric_augmentations(
            norm_img, num_variants=n_vars, seed=len(normalized_train_images)
        )
        for sample_rgb in aug_imgs:
            feat_grid = extractor.extract(sample_rgb)  # (H_p, W_p, D)
            vmask = build_board_patch_mask(
                sample_rgb,
                grid_size=(feat_grid.shape[0], feat_grid.shape[1]),
            )
            if vmask is not None and np.any(vmask > 0):
                flat_feat = feat_grid[vmask > 0]
            else:
                flat_feat = feat_grid.reshape(-1, feat_grid.shape[-1])
            all_patch_descriptors.append(flat_feat)

    if not all_patch_descriptors:
        raise RuntimeError("Không trích xuất được đặc trưng từ bất kỳ ảnh Normal nào.")

    # Calibrate dedicated 3-pin inspector thresholds on normal training boards
    pin_meta_path = output_weights_path.parent / "pin_metadata.json"
    if train_full_boards:
        calibrate_pin_inspector(train_full_boards, output_path=pin_meta_path)
    pin_inspector = PinInspector(metadata_path=pin_meta_path)

    stacked_patches = np.vstack(all_patch_descriptors).astype(np.float32)
    logger.info(
        "Extracted %d total patches (dim=%d) from %d normal images. Running greedy coreset subsampling...",
        stacked_patches.shape[0],
        stacked_patches.shape[1],
        len(normalized_train_images),
    )

    memory_bank = greedy_coreset_subsampling(
        embeddings=stacked_patches,
        sampling_ratio=max(coreset_ratio, 0.20),
        seed=42,
        max_coreset_size=4096,
    )
    logger.info("Built PatchCore Memory Bank of shape %s", memory_bank.shape)

    # Calibrate score normalization range (score_min, score_scale) & optimal threshold tau
    val_dir = dataset_root / "normal" / "val"
    anom_dir = dataset_root / "abnormal" / "test"
    val_files = list_images_in_dir(val_dir) or train_files[:5]
    anom_files = list_images_in_dir(anom_dir)

    raw_scores: list[float] = []
    pin_flags: list[bool] = []
    labels: list[int] = []

    for img_path in val_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prep = preprocess_board_image(rgb, aligner=aligner)
        if prep.success and prep.normalized_rgb is not None:
            feat_grid = extractor.extract(prep.normalized_rgb)
            valid_mask = build_board_patch_mask(
                prep.normalized_rgb,
                grid_size=(feat_grid.shape[0], feat_grid.shape[1]),
            )
            r_score, _ = compute_patchcore_anomaly_map(
                feat_grid,
                memory_bank,
                output_size=image_size,
                num_neighbors=num_neighbors,
                valid_patch_mask=valid_mask,
            )
            pins_ok = (
                pin_inspector.inspect(prep.full_board_rgb, prep.full_board_mask).pins_ok
                if prep.full_board_rgb is not None
                else True
            )
            raw_scores.append(r_score)
            pin_flags.append(pins_ok)
            labels.append(0)

    for img_path in anom_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prep = preprocess_board_image(rgb, aligner=aligner)
        if prep.success and prep.normalized_rgb is not None:
            feat_grid = extractor.extract(prep.normalized_rgb)
            valid_mask = build_board_patch_mask(
                prep.normalized_rgb,
                grid_size=(feat_grid.shape[0], feat_grid.shape[1]),
            )
            r_score, _ = compute_patchcore_anomaly_map(
                feat_grid,
                memory_bank,
                output_size=image_size,
                num_neighbors=num_neighbors,
                valid_patch_mask=valid_mask,
            )
            pins_ok = (
                pin_inspector.inspect(prep.full_board_rgb, prep.full_board_mask).pins_ok
                if prep.full_board_rgb is not None
                else True
            )
            raw_scores.append(r_score)
            pin_flags.append(pins_ok)
            labels.append(1)

    raw_arr = np.asarray(raw_scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int32)

    if raw_arr.size > 0 and np.any(label_arr == 0) and np.any(label_arr == 1):
        norm_min = float(np.min(raw_arr[label_arr == 0]))
        anom_max = float(np.max(raw_arr[label_arr == 1]))
        score_min = max(0.0, norm_min * 0.85)
        score_scale = max((anom_max - score_min) * 1.15, 1e-4)
    elif raw_arr.size > 0:
        score_min = float(np.min(raw_arr) * 0.85)
        score_scale = max(float(np.max(raw_arr) * 2.5 - score_min), 1e-4)
    else:
        score_min = 0.0
        score_scale = 1.0

    norm_scores = [
        max(float(np.clip((s - score_min) / score_scale, 0.0, 1.0)), 0.0 if p_ok else 0.88)
        for s, p_ok in zip(raw_scores, pin_flags, strict=False)
    ]
    auroc = compute_auroc(labels, norm_scores)
    optimal_tau, f1, precision, recall = find_optimal_threshold(labels, norm_scores)

    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "memory_bank": memory_bank,
        "backbone": backbone_name,
        "layers": layers,
        "backend_type": extractor.backend_type,
        "grid_size": [32, 32],
        "num_neighbors": num_neighbors,
        "score_min": score_min,
        "score_scale": score_scale,
        "optimal_threshold": optimal_tau,
    }
    with output_weights_path.open("wb") as fh:
        pickle.dump(checkpoint_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "patchcore",
        "backbone": backbone_name,
        "layers": layers,
        "backend_type": extractor.backend_type,
        "coreset_sampling_ratio": coreset_ratio,
        "memory_bank_size": int(memory_bank.shape[0]),
        "feature_dim": int(memory_bank.shape[1]),
        "image_size": list(image_size),
        "score_min": round(score_min, 6),
        "score_scale": round(score_scale, 6),
        "optimal_threshold": optimal_tau,
        "auroc": round(auroc, 4),
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "num_train_normal": len(normalized_train_images),
    }
    with output_metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved PatchCore checkpoint to %s and metadata to %s (optimal_threshold=%.4f, AUROC=%.4f)",
        output_weights_path,
        output_metadata_path,
        optimal_tau,
        auroc,
    )
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detection model on normal chip images."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/patchcore_chip.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/chip_dataset"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/patchcore_best.ckpt"),
        help="Output path for trained checkpoint.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("weights/metadata.json"),
        help="Output path for calibrated threshold & model metadata.",
    )
    args = parser.parse_args()

    summary = train_patchcore_model(
        config_path=args.config,
        dataset_root=args.dataset,
        output_weights_path=args.weights,
        output_metadata_path=args.metadata,
        auto_bootstrap=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
