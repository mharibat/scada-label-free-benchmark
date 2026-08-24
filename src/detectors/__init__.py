"""
Detector registry.

Each factory returns a fresh detector for a given random seed. `applicable`
lists which detectors run on each dataset (LSTM-AE and CNN-AE need sequential
windows, so they are BATADAL-only).
"""
from __future__ import annotations

from .autoencoders import (
    CNNAEDetector,
    CompactDenseAEDetector,
    DenseAEDetector,
    LSTMAEDetector,
    TransformerAEDetector,
)
from .classical import (
    IsolationForestDetector,
    MahalanobisDetector,
    OCSVMDetector,
    PCAReconstructionDetector,
)


def make_detector(key: str, seed: int = 42, epochs: int | None = None, **over):
    key = key.lower()
    if key == "mahalanobis":
        return MahalanobisDetector()
    if key in ("ocsvm", "oc-svm"):
        return OCSVMDetector(nu=0.05, fit_subset=5000, seed=seed)
    if key in ("isoforest", "iforest", "isolation_forest"):
        return IsolationForestDetector(n_estimators=200, contamination=0.05, seed=seed)
    if key in ("pca", "pca_reconstruction"):
        return PCAReconstructionDetector(variance=0.95, seed=seed)
    if key == "lstm_ae":
        return LSTMAEDetector(seed=seed, epochs=epochs or 40, batch_size=128, **over)
    if key == "dense_ae":            # windowed Dense-AE (BATADAL / HAI)
        return DenseAEDetector(seed=seed, epochs=epochs or 40, batch_size=128, dims=(256, 128, 64), **over)
    if key == "dense_ae_compact":    # tabular Gas-Pipeline Dense-AE (7,521 params)
        return CompactDenseAEDetector(seed=seed, epochs=epochs or 30, batch_size=512, **over)
    if key == "cnn_ae":
        return CNNAEDetector(seed=seed, epochs=epochs or 40, batch_size=128, **over)
    if key == "transformer_ae":
        return TransformerAEDetector(seed=seed, epochs=epochs or 30, batch_size=128, **over)
    raise ValueError(f"Unknown detector '{key}'")


# Which detectors run on which dataset, and the display label.
APPLICABLE = {
    "batadal": [
        ("mahalanobis", "Mahalanobis Distance"),
        ("pca", "PCA Reconstruction"),
        ("lstm_ae", "LSTM-Autoencoder"),
        ("dense_ae", "Dense-Autoencoder"),
        ("cnn_ae", "CNN-Autoencoder"),
        ("ocsvm", "One-Class SVM"),
        ("isoforest", "Isolation Forest"),
    ],
    "gas_pipeline": [
        ("mahalanobis", "Mahalanobis Distance"),
        ("pca", "PCA Reconstruction"),
        ("dense_ae_compact", "Dense-Autoencoder"),
        ("ocsvm", "One-Class SVM"),
        ("isoforest", "Isolation Forest"),
    ],
    "hai": [
        ("mahalanobis", "Mahalanobis Distance"),
        ("pca", "PCA Reconstruction"),
        ("lstm_ae", "LSTM-Autoencoder"),
        ("dense_ae", "Dense-Autoencoder"),
        ("cnn_ae", "CNN-Autoencoder"),
        ("transformer_ae", "Transformer-Autoencoder"),
        ("ocsvm", "One-Class SVM"),
        ("isoforest", "Isolation Forest"),
    ],
}

# Deterministic detectors don't vary across seeds -> run once.
DETERMINISTIC = {"mahalanobis", "pca"}

__all__ = ["make_detector", "APPLICABLE", "DETERMINISTIC"]
