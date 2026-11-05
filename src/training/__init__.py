"""Training and evaluation package for Chip Anomaly Detection System (Dev 2)."""

from typing import Any

__all__ = [
    "bootstrap_dataset_from_samples",
    "compute_auroc",
    "evaluate_dataset",
    "find_optimal_threshold",
    "generate_photometric_augmentations",
    "greedy_coreset_subsampling",
    "train_patchcore_model",
]


def __getattr__(name: str) -> Any:
    if name in ("compute_auroc", "evaluate_dataset", "find_optimal_threshold"):
        from src.training import evaluate

        return getattr(evaluate, name)
    if name in (
        "bootstrap_dataset_from_samples",
        "generate_photometric_augmentations",
        "greedy_coreset_subsampling",
        "train_patchcore_model",
    ):
        from src.training import train_patchcore

        return getattr(train_patchcore, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
