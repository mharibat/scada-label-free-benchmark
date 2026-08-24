"""
Statistical / classical one-class detectors.

Common interface:
    d = Detector(**hp)
    d.fit(X_train_normal)         # X: (N, W, F) windowed OR (N, F) tabular
    scores = d.score(X)           # higher score  =>  more anomalous

Windowed 3-D input is flattened to (N, W*F) for these tabular learners.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.linalg import pinvh
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def _flat(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


class MahalanobisDetector:
    """
    Parameter-free Gaussian model. Anomaly score = Mahalanobis distance to the
    training-set mean using the (regularised) pseudo-inverse covariance.
    """

    name = "Mahalanobis"

    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.mu_ = None
        self.VI_ = None
        self.fit_time_ = 0.0

    def fit(self, X):
        t0 = time.time()
        Xf = _flat(X).astype(np.float64)
        self.mu_ = Xf.mean(axis=0)
        cov = np.cov(Xf, rowvar=False)
        cov += self.reg * np.eye(cov.shape[0])
        self.VI_ = pinvh(cov)               # robust pseudo-inverse
        self.fit_time_ = time.time() - t0
        return self

    def score(self, X):
        Xf = _flat(X).astype(np.float64)
        d = Xf - self.mu_
        # squared Mahalanobis distance, vectorised
        m = np.einsum("ij,jk,ik->i", d, self.VI_, d)
        return np.sqrt(np.maximum(m, 0.0))


class OCSVMDetector:
    """One-Class SVM (RBF). Fit on a random subset for tractability."""

    name = "One-Class SVM"

    def __init__(self, nu: float = 0.05, gamma="scale", fit_subset: int = 5000, seed: int = 42):
        self.nu = nu
        self.gamma = gamma
        self.fit_subset = fit_subset
        self.seed = seed
        self.model = None
        self.fit_time_ = 0.0

    def fit(self, X):
        t0 = time.time()
        Xf = _flat(X)
        if self.fit_subset and len(Xf) > self.fit_subset:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(len(Xf), self.fit_subset, replace=False)
            Xf = Xf[idx]
        self.model = OneClassSVM(nu=self.nu, kernel="rbf", gamma=self.gamma)
        self.model.fit(Xf)
        self.fit_time_ = time.time() - t0
        return self

    def score(self, X):
        # decision_function: >0 inlier, <0 outlier -> negate so higher = anomaly
        return -self.model.decision_function(_flat(X))


class IsolationForestDetector:
    """Isolation Forest. Score = negative average path length (higher=anomaly)."""

    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, seed: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self.model = None
        self.fit_time_ = 0.0

    def fit(self, X):
        t0 = time.time()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(_flat(X))
        self.fit_time_ = time.time() - t0
        return self

    def score(self, X):
        return -self.model.score_samples(_flat(X))


class PCAReconstructionDetector:
    """PCA reconstruction error with variance-retaining component selection."""

    name = "PCA Reconstruction"

    def __init__(self, variance: float = 0.95, max_fit: int = 50000, seed: int = 42):
        self.variance = variance
        self.max_fit = max_fit
        self.seed = seed
        self.model = None
        self.fit_time_ = 0.0
        self.n_params = 0

    def fit(self, X):
        t0 = time.time()
        Xf = _flat(X).astype(np.float64)
        if self.max_fit and len(Xf) > self.max_fit:
            rng = np.random.RandomState(self.seed)
            Xf = Xf[rng.choice(len(Xf), self.max_fit, replace=False)]
        self.model = PCA(n_components=self.variance, svd_solver="full")
        self.model.fit(Xf)
        self.n_params = int(self.model.components_.size + self.model.mean_.size)
        self.fit_time_ = time.time() - t0
        return self

    def score(self, X):
        Xf = _flat(X).astype(np.float64)
        rec = self.model.inverse_transform(self.model.transform(Xf))
        return np.mean((Xf - rec) ** 2, axis=1)
