"""
SCADA one-class anomaly-detection benchmark (BATADAL + Gas Pipeline).

Public API used by scripts and notebooks.
"""
from .utils import set_seed, ensure_dir, load_config, mean_std
from .metrics import binary_metrics, best_threshold_by_f1, f1_point_adjust, bootstrap_ci
from .dataset import OneClassDataset
from .loaders import build_batadal, build_gas_pipeline, build_dataset
from .detectors import make_detector, APPLICABLE, DETERMINISTIC
from .experiment import run_benchmark, run_detector

__all__ = [
    "set_seed", "ensure_dir", "load_config", "mean_std",
    "binary_metrics", "best_threshold_by_f1", "f1_point_adjust", "bootstrap_ci",
    "OneClassDataset",
    "build_batadal", "build_gas_pipeline", "build_dataset",
    "make_detector", "APPLICABLE", "DETERMINISTIC",
    "run_benchmark", "run_detector",
]
