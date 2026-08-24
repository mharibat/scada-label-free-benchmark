"""
Reconstruction-based autoencoder detectors (PyTorch, CPU-friendly).

All are trained ONE-CLASS on normal data only; the anomaly score of a sample
is its reconstruction error (mean squared error). A small held-out slice of the
NORMAL training data is used for early stopping, so anomalies never leak into
model selection.

Architectures
-------------
LSTM-AE   : sequence encoder-decoder for windowed input (N, W, F).
DenseAE   : fully-connected AE on the flattened window (N, W*F)   [BATADAL], or
            compact 17-64-32-16-32-64-17 AE on tabular input (N, 17)  [Gas].
CNN-AE    : 1-D convolutional AE for windowed input (N, W, F).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))


def _count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
class _LSTMAE(nn.Module):
    def __init__(self, n_features: int, window: int, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.window = window
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc2lat = nn.Linear(hidden, latent)
        self.lat2dec = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):                       # x: (B, W, F)
        _, (h, _) = self.enc(x)
        z = self.enc2lat(h[-1])                 # (B, latent)
        d = self.lat2dec(z).unsqueeze(1).repeat(1, self.window, 1)
        d, _ = self.dec(d)
        return self.out(d)


class _DenseAE(nn.Module):
    def __init__(self, n_in: int, dims=(256, 128, 64)):
        super().__init__()
        d1, d2, d3 = dims
        self.net = nn.Sequential(
            nn.Linear(n_in, d1), nn.ReLU(),
            nn.Linear(d1, d2), nn.ReLU(),
            nn.Linear(d2, d3), nn.ReLU(),
            nn.Linear(d3, d2), nn.ReLU(),
            nn.Linear(d2, d1), nn.ReLU(),
            nn.Linear(d1, n_in), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class _CNNAE(nn.Module):
    def __init__(self, n_features: int, window: int, ch: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch // 2, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(ch // 2, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, n_features, 3, padding=1),
        )

    def forward(self, x):                       # x: (B, W, F)
        x = x.transpose(1, 2)                   # (B, F, W)
        z = self.enc(x)
        r = self.dec(z)
        return r.transpose(1, 2)                # (B, W, F)


class _TransformerAE(nn.Module):
    """Compact sequence-to-sequence Transformer reconstruction model."""

    def __init__(self, n_features: int, window: int, d_model: int = 32,
                 nhead: int = 4, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.out_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        h = self.in_proj(x) + self.pos[:, :x.shape[1]]
        return self.out_proj(self.encoder(h))


# --------------------------------------------------------------------------- #
# Trainer wrapper (shared)
# --------------------------------------------------------------------------- #
class _AEDetector:
    name = "Autoencoder"
    kind = "dense"          # 'dense' | 'lstm' | 'cnn'

    def __init__(self, epochs=40, batch_size=256, lr=1e-3, patience=6,
                 seed=42, verbose=False, **kw):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.kw = kw
        self.model: Optional[nn.Module] = None
        self.n_params = 0
        self.fit_time_ = 0.0

    # -- to be overridden --
    def _build(self, X) -> nn.Module:
        raise NotImplementedError

    def _prep(self, X) -> torch.Tensor:
        """Native -> tensor in the shape the model expects."""
        if self.kind == "dense":
            X = X.reshape(X.shape[0], -1)
        return torch.as_tensor(np.asarray(X, dtype=np.float32))

    def fit(self, X):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        t0 = time.time()
        T = self._prep(X)
        # internal normal-only validation slice for early stopping
        n = len(T)
        n_val = max(1, int(n * 0.1))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Ttr, Tval = T[tr_idx], T[val_idx]

        self.model = self._build(X)
        self.n_params = _count_params(self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best, best_state, wait = float("inf"), None, 0
        for ep in range(self.epochs):
            self.model.train()
            idx = torch.randperm(len(Ttr))
            for i in range(0, len(Ttr), self.batch_size):
                b = Ttr[idx[i:i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model(b), b)
                loss.backward()
                opt.step()
            # validation reconstruction loss
            self.model.eval()
            with torch.no_grad():
                vloss = loss_fn(self.model(Tval), Tval).item()
            if self.verbose:
                print(f"    epoch {ep+1:>3} val_mse={vloss:.6f}")
            if vloss < best - 1e-6:
                best, best_state, wait = vloss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fit_time_ = time.time() - t0
        return self

    def score(self, X):
        self.model.eval()
        T = self._prep(X)
        errs = np.empty(len(T), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(T), 1024):
                b = T[i:i + 1024]
                r = self.model(b)
                e = ((r - b) ** 2).reshape(len(b), -1).mean(dim=1)
                errs[i:i + len(b)] = e.numpy()
        return errs


# --------------------------------------------------------------------------- #
# Concrete detectors
# --------------------------------------------------------------------------- #
class LSTMAEDetector(_AEDetector):
    name = "LSTM-Autoencoder"
    kind = "lstm"

    def _build(self, X):
        _, W, F = X.shape
        return _LSTMAE(F, W, hidden=self.kw.get("hidden", 64), latent=self.kw.get("latent", 32))


class DenseAEDetector(_AEDetector):
    name = "Dense-Autoencoder"
    kind = "dense"

    def _build(self, X):
        n_in = int(np.prod(X.shape[1:]))
        dims = self.kw.get("dims", (256, 128, 64))
        return _DenseAE(n_in, dims=dims)


class CompactDenseAEDetector(_AEDetector):
    """17-64-32-16-32-64-17 tabular AE (exactly 7,521 params) for Gas Pipeline."""
    name = "Dense-Autoencoder"
    kind = "dense"

    def _build(self, X):
        n_in = int(np.prod(X.shape[1:]))
        return _DenseAE(n_in, dims=(64, 32, 16))


class CNNAEDetector(_AEDetector):
    name = "CNN-Autoencoder"
    kind = "cnn"

    def _build(self, X):
        _, W, F = X.shape
        return _CNNAE(F, W, ch=self.kw.get("ch", 64))


class TransformerAEDetector(_AEDetector):
    name = "Transformer-Autoencoder"
    kind = "transformer"

    def _build(self, X):
        _, W, F = X.shape
        return _TransformerAE(
            F, W, d_model=self.kw.get("d_model", 32),
            nhead=self.kw.get("nhead", 4), layers=self.kw.get("layers", 2),
        )
