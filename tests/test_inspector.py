"""Unit and contract tests for Dev 2: PatchCore Inspector, Training & Evaluation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference.inspector import (
    InspectionModel,
    build_heatmap_overlay,
    compute_patchcore_anomaly_map,
    update_threshold_decision,
)
from src.schemas import InspectionResult, PreprocessResult
from src.training.evaluate import (
    compute_auroc,
    evaluate_dataset,
    find_optimal_threshold,
)
from src.training.train_patchcore import (
    generate_photometric_augmentations,
    greedy_coreset_subsampling,
    train_patchcore_model,
)


class DummyAlignerSuccess:
    """Stub aligner returning a valid 256x256 normalized board image."""

    def process(self, image_rgb: np.ndarray) -> PreprocessResult:
        h, w = 256, 256
        cropped = np.full((300, 100, 3), 120, dtype=np.uint8)
        normalized = np.full((h, w, 3), 128, dtype=np.uint8)
        box = np.array([[10, 10], [110, 10], [110, 310], [10, 310]], dtype=np.int32)
        return PreprocessResult(
            success=True,
            cropped_rgb=cropped,
            normalized_rgb=normalized,
            bbox_box_points=box,
            rotation_angle=0.0,
        )


class DummyAlignerNoBoard:
    """Stub aligner simulating an empty background with no board detected."""

    def process(self, image_rgb: np.ndarray) -> PreprocessResult:
        return PreprocessResult(
            success=False,
            error_message="Không tìm thấy bo mạch hợp lệ trong khung hình",
        )


class StubInspectorFixedScore(InspectionModel):
    """Subclass overriding _run_patchcore to return a deterministic anomaly score=0.62."""

    def _run_patchcore(
        self, normalized_rgb: np.ndarray, cropped_rgb: np.ndarray | None = None
    ) -> tuple[float, np.ndarray, np.ndarray]:
        anomaly_map = np.zeros((256, 256), dtype=np.float32)
        anomaly_map[100:140, 100:140] = 0.85
        overlay = build_heatmap_overlay(normalized_rgb, anomaly_map)
        return 0.62, anomaly_map, overlay


def test_inspector_threshold_decision_contract(tmp_path: Path) -> None:
    """Verify threshold comparison: score=0.62 is Normal at tau=0.70 and Anomaly at tau=0.50."""
    dummy_weights = tmp_path / "patchcore_best.ckpt"
    dummy_weights.write_bytes(b"placeholder")

    inspector = StubInspectorFixedScore(
        weights_path=dummy_weights,
        default_threshold=0.50,
        aligner=DummyAlignerSuccess(),
    )
    raw_img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Case 1: threshold = 0.70 -> score 0.62 <= 0.70 -> is_normal == True
    res_normal = inspector.predict(raw_img, threshold=0.70)
    assert isinstance(res_normal, InspectionResult)
    assert res_normal.status == "OK"
    assert res_normal.is_normal is True
    assert res_normal.anomaly_score == pytest.approx(0.62, abs=1e-4)
    assert res_normal.threshold == pytest.approx(0.70, abs=1e-4)
    assert res_normal.anomaly_map is not None
    assert res_normal.anomaly_map.shape == (256, 256)
    assert res_normal.heatmap_overlay_rgb is not None
    assert res_normal.heatmap_overlay_rgb.shape == (256, 256, 3)
    assert res_normal.processing_time_ms >= 0.0

    # Case 2: threshold = 0.50 -> score 0.62 > 0.50 -> is_normal == False
    res_anomaly = inspector.predict(raw_img, threshold=0.50)
    assert res_anomaly.status == "OK"
    assert res_anomaly.is_normal is False
    assert res_anomaly.anomaly_score == pytest.approx(0.62, abs=1e-4)
    assert res_anomaly.threshold == pytest.approx(0.50, abs=1e-4)


def test_inspector_handles_no_board_detected(tmp_path: Path) -> None:
    """Verify inspector returns NO_BOARD_DETECTED without running PatchCore when aligner fails."""
    dummy_weights = tmp_path / "patchcore_best.ckpt"
    dummy_weights.write_bytes(b"placeholder")

    inspector = StubInspectorFixedScore(
        weights_path=dummy_weights,
        default_threshold=0.55,
        aligner=DummyAlignerNoBoard(),
    )
    blank_img = np.full((400, 400, 3), 255, dtype=np.uint8)

    result = inspector.predict(blank_img)
    assert result.status == "NO_BOARD_DETECTED"
    assert result.is_normal is None
    assert result.anomaly_score is None
    assert result.threshold == pytest.approx(0.55, abs=1e-4)
    assert "Không tìm thấy bo mạch" in result.message


def test_inspector_load_raises_when_weights_missing(tmp_path: Path) -> None:
    """Verify load() raises FileNotFoundError when checkpoint file does not exist."""
    missing_path = tmp_path / "nonexistent.ckpt"
    inspector = InspectionModel(
        weights_path=missing_path,
        aligner=DummyAlignerSuccess(),
    )
    with pytest.raises(FileNotFoundError):
        inspector.load()


def test_update_threshold_decision_live_slider() -> None:
    """Verify live threshold slider helper flips is_normal without re-running model."""
    initial = InspectionResult(
        status="OK",
        is_normal=False,
        anomaly_score=0.58,
        threshold=0.50,
        processing_time_ms=25.0,
        message="Anomaly (Phát hiện bất thường)",
    )
    updated = update_threshold_decision(initial, new_threshold=0.65)
    assert updated.is_normal is True
    assert updated.threshold == pytest.approx(0.65)
    assert updated.anomaly_score == pytest.approx(0.58)
    assert "Normal" in updated.message


def test_patchcore_memory_bank_knn_scoring_detects_outlier_patch() -> None:
    """Verify PatchCore k-NN scoring assigns low score to normal patches and high score to anomalous patch."""
    rng = np.random.default_rng(42)
    # Memory bank with 64 normal patch descriptors of dimension 32
    memory_bank = rng.normal(loc=0.0, scale=0.1, size=(64, 32)).astype(np.float32)

    # Normal feature grid (16x16 patches, dim=32) drawn from same distribution
    normal_patches = rng.normal(loc=0.0, scale=0.1, size=(16, 16, 32)).astype(np.float32)
    score_norm, map_norm = compute_patchcore_anomaly_map(
        patch_features=normal_patches,
        memory_bank=memory_bank,
        output_size=(256, 256),
        num_neighbors=5,
    )

    # Anomalous feature grid with a strong local defect in the center (missing SMD component)
    anomaly_patches = normal_patches.copy()
    anomaly_patches[7:9, 7:9, :] += 5.0
    score_anom, map_anom = compute_patchcore_anomaly_map(
        patch_features=anomaly_patches,
        memory_bank=memory_bank,
        output_size=(256, 256),
        num_neighbors=5,
    )

    assert map_norm.shape == (256, 256)
    assert map_anom.shape == (256, 256)
    assert score_anom > score_norm * 3.0
    # Peak of anomaly map should be near the center (around y=112..144, x=112..144)
    peak_y, peak_x = np.unravel_index(np.argmax(map_anom), map_anom.shape)
    assert 80 <= peak_y <= 176
    assert 80 <= peak_x <= 176


def test_greedy_coreset_subsampling_reduces_memory_bank_size() -> None:
    """Verify greedy k-center coreset subsampling returns exact requested ratio and preserves dimension."""
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(200, 24)).astype(np.float32)
    coreset = greedy_coreset_subsampling(embeddings, sampling_ratio=0.15, seed=123)

    assert coreset.shape == (30, 24)
    assert coreset.dtype == np.float32


def test_photometric_augmentation_preserves_board_orientation_no_flip() -> None:
    """Verify photometric augmentation alters lighting but never flips asymmetric board features."""
    # Create an asymmetric test image: bright marker exclusively in top-left quadrant
    base_img = np.full((256, 256, 3), 40, dtype=np.uint8)
    base_img[20:60, 20:60] = 230

    aug_list = generate_photometric_augmentations(base_img, num_variants=4, seed=7)
    assert len(aug_list) == 4

    for aug in aug_list:
        assert aug.shape == (256, 256, 3)
        assert aug.dtype == np.uint8
        # Top-left quadrant must remain significantly brighter than bottom-right and top-right
        top_left_mean = float(np.mean(aug[20:60, 20:60]))
        top_right_mean = float(np.mean(aug[20:60, 196:236]))
        bottom_left_mean = float(np.mean(aug[196:236, 20:60]))
        assert top_left_mean > top_right_mean + 80.0
        assert top_left_mean > bottom_left_mean + 80.0


def test_compute_auroc_and_find_optimal_threshold() -> None:
    """Verify AUROC and optimal threshold tau calibration on separated Normal vs Anomaly scores."""
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.12, 0.18, 0.22, 0.25, 0.29, 0.61, 0.68, 0.74, 0.82, 0.91]

    auroc = compute_auroc(labels, scores)
    tau, f1, precision, recall = find_optimal_threshold(labels, scores)

    assert auroc == pytest.approx(1.0, abs=1e-5)
    assert 0.29 <= tau < 0.61
    assert f1 == pytest.approx(1.0, abs=1e-5)
    assert precision == pytest.approx(1.0, abs=1e-5)
    assert recall == pytest.approx(1.0, abs=1e-5)


def test_end_to_end_trained_inspector_on_real_samples(tmp_path: Path) -> None:
    """Verify trained PatchCore checkpoint classifies Normal validation sample as True and Anomaly sample as False."""
    import shutil

    weights_path = Path("weights/patchcore_best.ckpt")
    metadata_path = Path("weights/metadata.json")

    val_normal_candidates = sorted(Path("data/chip_dataset/normal/val").glob("*.jpg"))
    test_anom_candidates = sorted(Path("data/chip_dataset/abnormal/test").glob("*.jpg"))

    if Path("samples/normal_val_001.jpg").exists():
        sample_normal = Path("samples/normal_val_001.jpg")
    elif val_normal_candidates:
        sample_normal = val_normal_candidates[0]
    else:
        sample_normal = Path("image/sample2.jpg")

    # Prefer anomaly_test_005.jpg (taped 103 potentiometer/SMD zone) when available
    if Path("samples/anomaly_test_005.jpg").exists():
        sample_anomaly = Path("samples/anomaly_test_005.jpg")
    elif Path("data/chip_dataset/abnormal/test/anomaly_test_005.jpg").exists():
        sample_anomaly = Path("data/chip_dataset/abnormal/test/anomaly_test_005.jpg")
    elif test_anom_candidates:
        sample_anomaly = test_anom_candidates[0]
    else:
        sample_anomaly = Path("image/sample1.jpg")

    if not weights_path.exists():
        train_patchcore_model(
            output_weights_path=weights_path,
            output_metadata_path=metadata_path,
            auto_bootstrap=True,
        )

    inspector = InspectionModel(
        weights_path=weights_path,
        metadata_path=metadata_path,
    )
    inspector.load()

    norm_bgr = cv2.imread(str(sample_normal))
    anom_bgr = cv2.imread(str(sample_anomaly))
    assert norm_bgr is not None
    assert anom_bgr is not None

    norm_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
    anom_rgb = cv2.cvtColor(anom_bgr, cv2.COLOR_BGR2RGB)

    res_norm = inspector.predict(norm_rgb)
    res_anom = inspector.predict(anom_rgb)

    assert res_norm.status == "OK"
    assert res_anom.status == "OK"
    assert res_norm.is_normal is True
    assert res_anom.is_normal is False
    assert res_norm.anomaly_score is not None
    assert res_anom.anomaly_score is not None
    assert res_anom.anomaly_score > res_norm.anomaly_score
    assert res_anom.anomaly_map is not None
    assert res_anom.heatmap_overlay_rgb is not None
    assert res_anom.heatmap_overlay_rgb.shape == (256, 256, 3)

    # Verify heatmap peak on anomaly sample is inside the active PCB component zone (around 103 pot / SMD),
    # not on the top/bottom cut edges or black letterbox background
    peak_y, peak_x = np.unravel_index(np.argmax(res_anom.anomaly_map), res_anom.anomaly_map.shape)
    assert 65 <= int(peak_y) <= 195, f"Expected peak_y in PCB component zone [65..195], got {peak_y}"
    assert 98 <= int(peak_x) <= 158, f"Expected peak_x in PCB width [98..158], got {peak_x}"

    if val_normal_candidates and test_anom_candidates:
        eval_root = Path("data/chip_dataset")
    else:
        eval_root = tmp_path / "mini_eval_dataset"
        norm_dir = eval_root / "normal" / "val"
        anom_dir = eval_root / "abnormal" / "test"
        norm_dir.mkdir(parents=True, exist_ok=True)
        anom_dir.mkdir(parents=True, exist_ok=True)
        for p in [*Path("samples").glob("normal_*.jpg"), Path("image/sample2.jpg")]:
            if p.is_file():
                shutil.copy2(p, norm_dir / p.name)
        for p in [*Path("samples").glob("anomaly_*.jpg"), Path("image/sample1.jpg")]:
            if p.is_file():
                shutil.copy2(p, anom_dir / p.name)

    report = evaluate_dataset(
        weights_path=weights_path,
        dataset_root=eval_root,
        metadata_path=metadata_path,
        update_metadata=False,
    )
    assert report["auroc"] >= 0.92
    assert report["f1_score"] >= 0.90


def _make_synthetic_full_board(
    missing_pin_idx: int | None = None,
    shorten_pin_idx: int | None = None,
    bend_pin_idx: int | None = None,
) -> np.ndarray:
    """Create a synthetic (640, 144, 3) full_board_rgb with dark PCB top and 3 metallic pins at bottom."""
    canvas = np.full((640, 144, 3), 220, dtype=np.uint8)
    # Dark PCB body from y=16 to y=528
    canvas[16:528, 12:132] = 28
    # Pin root x positions in 144 width (corresponding to ~86, 128, 170 in 256 width)
    xs = [48, 72, 96]
    for idx, x0 in enumerate(xs):
        if missing_pin_idx == idx:
            continue
        pin_len = 82
        if shorten_pin_idx == idx:
            pin_len = int(82 * 0.45)  # shortened by >50%
        y_start = 528
        y_end = y_start + pin_len
        if bend_pin_idx == idx:
            # Draw a pin bent at ~24 degrees
            pt1 = (x0, y_start)
            pt2 = (x0 - 34, y_end)
            cv2.line(canvas, pt1, pt2, (75, 75, 75), thickness=5, lineType=cv2.LINE_AA)
        else:
            cv2.line(canvas, (x0, y_start), (x0, y_end), (75, 75, 75), thickness=5, lineType=cv2.LINE_AA)
    return canvas


def test_pin_inspector_synthetic_normal_missing_short_and_bent_pins(tmp_path: Path) -> None:
    """Verify PinInspector detects normal 3 pins, missing pin, shortened pin, and bent pin."""
    from src.inference.pin_inspector import PinInspector

    pin_insp = PinInspector(metadata_path=tmp_path / "nonexistent_pin_meta.json")

    # 1. Normal 3 straight pins -> OK
    board_ok = _make_synthetic_full_board()
    res_ok = pin_insp.inspect(board_ok)
    assert res_ok.pins_ok is True, f"Expected synthetic normal pins OK, got: {res_ok.message}"
    assert res_ok.pin_count == 3
    assert len(res_ok.per_pin) == 3
    assert res_ok.pin_roi_rgb is not None and res_ok.pin_roi_rgb.shape == (256, 256, 3)
    assert res_ok.overlay_rgb is not None and res_ok.overlay_rgb.shape == (256, 256, 3)

    # 2. Missing middle pin -> FAIL
    board_missing = _make_synthetic_full_board(missing_pin_idx=1)
    res_missing = pin_insp.inspect(board_missing)
    assert res_missing.pins_ok is False
    assert res_missing.pin_count == 2
    assert "thiếu" in res_missing.message.lower()

    # 3. Shortened pin (cut by >40%) -> FAIL
    board_short = _make_synthetic_full_board(shorten_pin_idx=2)
    res_short = pin_insp.inspect(board_short)
    assert res_short.pins_ok is False
    assert "ngắn" in res_short.message.lower() or "thiếu" in res_short.message.lower()

    # 4. Bent pin (~24 degrees) -> FAIL
    board_bent = _make_synthetic_full_board(bend_pin_idx=0)
    res_bent = pin_insp.inspect(board_bent)
    assert res_bent.pins_ok is False
    assert "cong" in res_bent.message.lower()


def test_pin_inspector_on_real_bent_and_taped_pin_samples() -> None:
    """Verify PinInspector and InspectionModel catch real pin anomalies (bent & taped pins)."""
    from src.inference.pin_inspector import PinInspector
    from src.preprocessing.board_aligner import create_board_aligner, preprocess_board_image

    aligner = create_board_aligner()
    pin_insp = PinInspector()

    norm_path = Path("image/sample2.jpg")
    if norm_path.exists():
        norm_bgr = cv2.imread(str(norm_path))
        assert norm_bgr is not None
        prep_norm = preprocess_board_image(cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB), aligner=aligner)
        assert prep_norm.full_board_rgb is not None
        pin_res_norm = pin_insp.inspect(prep_norm.full_board_rgb)
        assert pin_res_norm.pins_ok is True
        assert pin_res_norm.pin_count == 3

    checked_defects = 0
    for candidates in (
        ("samples/anomaly_test_003.jpg", "data/chip_dataset/abnormal/test/anomaly_test_003.jpg"),
        ("data/chip_dataset/abnormal/test/anomaly_test_009.jpg",),
        ("samples/anomaly_test_011.jpg", "data/chip_dataset/abnormal/test/anomaly_test_011.jpg"),
    ):
        p = next((Path(c) for c in candidates if Path(c).exists()), None)
        if p is not None:
            bgr = cv2.imread(str(p))
            assert bgr is not None
            prep_def = preprocess_board_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), aligner=aligner)
            assert prep_def.full_board_rgb is not None
            pin_res_def = pin_insp.inspect(prep_def.full_board_rgb)
            assert pin_res_def.pins_ok is False, f"Expected pin defect on {p.name}, got OK"
            checked_defects += 1
    assert checked_defects >= 2, "Expected at least 2 real pin defect samples in samples/ or data/"


def test_patchcore_feature_extractor_offline_backbone_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PatchCoreFeatureExtractor loads local fp16 backbone when internet download fails."""
    import torch
    from torchvision import models

    from src.inference.inspector import PatchCoreFeatureExtractor

    orig_fn = models.resnet18
    base_model = orig_fn(weights=None)
    sd_fp16 = {
        k: (v.half() if v.is_floating_point() else v)
        for k, v in base_model.state_dict().items()
        if not k.startswith(("layer4.", "fc."))
    }
    offline_pt = tmp_path / "resnet18_layer3_fp16.pt"
    torch.save(sd_fp16, offline_pt)

    def _mock_resnet18(*args: object, **kwargs: object) -> object:
        if kwargs.get("weights") is not None:
            raise OSError("Simulated offline environment: network unreachable")
        return orig_fn(*args, **kwargs)

    monkeypatch.setattr(models, "resnet18", _mock_resnet18)

    extractor = PatchCoreFeatureExtractor(
        backbone_name="resnet18",
        layers=["layer2", "layer3"],
        offline_backbone_path=offline_pt,
    )
    assert extractor.backend_type == "torch_backbone"
    dummy_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
    feats = extractor.extract(dummy_rgb)
    assert feats.shape[2] == 384
    assert feats.dtype == np.float32





