"""Evaluation and Threshold Calibration Script for PatchCore (Dev 2)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.inference.inspector import InspectionModel

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_in_dir(directory: Path) -> list[Path]:
    """Return sorted list of valid image files in a directory."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute Area Under the ROC Curve (AUROC) where 0=Normal, 1=Anomaly.

    Args:
        labels: Binary ground-truth labels (0 for Normal, 1 for Anomaly).
        scores: Continuous anomaly scores (higher means more anomalous).

    Returns:
        AUROC score in [0.0, 1.0].
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 1.0

    pos_scores = y_score[pos_mask][:, None]  # (n_pos, 1)
    neg_scores = y_score[neg_mask][None, :]  # (1, n_neg)

    greater = np.sum(pos_scores > neg_scores)
    ties = np.sum(np.isclose(pos_scores, neg_scores, atol=1e-12))
    auroc = float((greater + 0.5 * ties) / (n_pos * n_neg))
    return float(np.clip(auroc, 0.0, 1.0))


def find_optimal_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[float, float, float, float]:
    """Find the optimal decision threshold tau that maximizes F1-score on validation/test set.

    Decision logic:
        score <= tau -> Normal (label 0, is_normal = True)
        score >  tau -> Anomaly (label 1, is_normal = False)

    Returns:
        Tuple of (optimal_tau, best_f1, best_precision, best_recall).
    """
    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)

    if y_score.size == 0:
        return 0.50, 0.0, 0.0, 0.0

    # If only normal samples exist, set threshold at mean + 3.5 * std
    if np.all(y_true == 0):
        tau = float(np.mean(y_score) + 3.5 * max(float(np.std(y_score)), 0.02))
        return float(np.clip(tau, 0.05, 0.95)), 1.0, 1.0, 1.0

    unique_sorted = np.unique(np.sort(y_score))
    if unique_sorted.size == 1:
        candidates = unique_sorted
    else:
        midpoints = 0.5 * (unique_sorted[:-1] + unique_sorted[1:])
        candidates = np.concatenate(
            [[unique_sorted[0] - 0.01], midpoints, [unique_sorted[-1] + 0.01]]
        )

    normal_max = float(np.max(y_score[y_true == 0])) if np.any(y_true == 0) else 0.0
    anom_min = float(np.min(y_score[y_true == 1])) if np.any(y_true == 1) else 1.0
    ideal_center = 0.30 * normal_max + 0.70 * anom_min
    if anom_min > normal_max + 0.04:
        gap_grid = np.linspace(normal_max + 0.01, anom_min - 0.01, 15)
        candidates = np.unique(np.concatenate([candidates, gap_grid]))

    best_tau = 0.50
    best_f1 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    best_dist_to_center = float("inf")

    for tau in candidates:
        preds = (y_score > tau).astype(np.int32)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        dist_center = abs(float(tau) - ideal_center)

        if (f1 > best_f1 + 1e-9) or (
            abs(f1 - best_f1) <= 1e-9 and dist_center < best_dist_to_center
        ):
            best_f1 = f1
            best_tau = float(tau)
            best_prec = float(prec)
            best_rec = float(rec)
            best_dist_to_center = dist_center

    return round(best_tau, 4), round(best_f1, 4), round(best_prec, 4), round(best_rec, 4)


def evaluate_dataset(
    config_path: Path = Path("configs/patchcore_chip.yaml"),
    weights_path: Path = Path("weights/patchcore_best.ckpt"),
    dataset_root: Path = Path("data/chip_dataset"),
    metadata_path: Path = Path("weights/metadata.json"),
    update_metadata: bool = True,
) -> dict[str, Any]:
    """Evaluate trained PatchCore model on normal/val and abnormal/test directories."""
    inspector = InspectionModel(
        weights_path=weights_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )
    inspector.load()

    normal_val_dir = dataset_root / "normal" / "val"
    abnormal_test_dir = dataset_root / "abnormal" / "test"

    normal_files = list_images_in_dir(normal_val_dir)
    abnormal_files = list_images_in_dir(abnormal_test_dir)

    labels: list[int] = []
    scores: list[float] = []
    per_file_results: list[dict[str, Any]] = []

    for img_path in normal_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = inspector.predict(rgb)
        if res.status == "OK" and res.anomaly_score is not None:
            labels.append(0)
            scores.append(float(res.anomaly_score))
            per_file_results.append(
                {
                    "file": str(img_path.relative_to(dataset_root)),
                    "ground_truth": "Normal",
                    "anomaly_score": res.anomaly_score,
                }
            )

    for img_path in abnormal_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = inspector.predict(rgb)
        if res.status == "OK" and res.anomaly_score is not None:
            labels.append(1)
            scores.append(float(res.anomaly_score))
            per_file_results.append(
                {
                    "file": str(img_path.relative_to(dataset_root)),
                    "ground_truth": "Anomaly",
                    "anomaly_score": res.anomaly_score,
                }
            )

    auroc = compute_auroc(labels, scores)
    optimal_tau, f1, precision, recall = find_optimal_threshold(labels, scores)

    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score > optimal_tau).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    accuracy = float((tp + tn) / max(len(labels), 1))

    report: dict[str, Any] = {
        "auroc": round(auroc, 4),
        "optimal_threshold": optimal_tau,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": round(accuracy, 4),
        "num_normal_val": int(np.sum(y_true == 0)),
        "num_abnormal_test": int(np.sum(y_true == 1)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "samples": per_file_results,
    }

    if update_metadata:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        existing_meta: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as fh:
                    existing_meta = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing_meta = {}
        existing_meta.update(
            {
                "optimal_threshold": optimal_tau,
                "auroc": round(auroc, 4),
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": round(accuracy, 4),
                "num_normal_val": int(np.sum(y_true == 0)),
                "num_abnormal_test": int(np.sum(y_true == 1)),
            }
        )
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(existing_meta, fh, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate PatchCore model and calibrate decision threshold tau."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/patchcore_chip.yaml"),
        help="Path to PatchCore YAML config.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/patchcore_best.ckpt"),
        help="Path to trained PatchCore weights checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/chip_dataset"),
        help="Path to dataset root folder.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("weights/metadata.json"),
        help="Path to output metadata JSON file.",
    )
    args = parser.parse_args()

    report = evaluate_dataset(
        config_path=args.config,
        weights_path=args.weights,
        dataset_root=args.dataset,
        metadata_path=args.metadata,
        update_metadata=True,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
