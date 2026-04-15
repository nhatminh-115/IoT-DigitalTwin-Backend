"""
Autoencoder-based anomaly detector.

Usage in api_service.py background loop:
    from .anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    result = detector.score(clean_df)
    if result.is_anomaly:
        # push Telegram alert
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model (must match training architecture exactly)
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, window: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, window * n_features),
        )
        self.window     = window
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self.encoder(x)
        out = self.decoder(z)
        return out.view(-1, self.window, self.n_features)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    is_anomaly:         bool
    reconstruction_error: float
    threshold:          float
    top_features:       list[tuple[str, float]]   # (feature_name, per-feature error)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Loads checkpoint + meta at init, exposes .score(df) for inference.
    Thread-safe for read (no mutable state after init).
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/autoencoder_checkpoint.pt",
        meta_path:       str | Path = "artifacts/autoencoder_meta.json",
    ):
        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_cols: list[str] = meta["feature_cols"]
        self.window:       int       = meta["window"]
        self.n_features:   int       = meta["n_features"]
        self.threshold:    float     = meta["threshold"]
        self._mean  = np.array(meta["scaler_mean"],  dtype=np.float32)
        self._scale = np.array(meta["scaler_scale"], dtype=np.float32)

        self._model = _Autoencoder(self.n_features, self.window)
        self._model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()
        logger.info(
            "AnomalyDetector ready: %s features, threshold=%.4f",
            self.n_features,
            self.threshold,
        )

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._mean) / self._scale

    def score(self, df) -> AnomalyResult | None:
        """
        Args:
            df: pd.DataFrame with DatetimeIndex and sensor columns.
                Must have at least `window` rows.

        Returns:
            AnomalyResult or None if not enough data / missing columns.
        """
        # Check which feature_cols are available
        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < self.n_features * 0.5:
            return None

        # Build input window from last `window` rows
        if len(df) < self.window:
            return None

        tail = df[available].tail(self.window).copy()

        # Fill missing columns with column mean (handles partial node outages)
        for col in self.feature_cols:
            if col not in tail.columns:
                tail[col] = 0.0

        tail = tail[self.feature_cols].ffill().bfill().fillna(0.0)
        arr  = tail.values.astype(np.float32)
        arr  = self._normalize(arr)

        x    = torch.tensor(arr).unsqueeze(0)   # (1, window, n_features)

        with torch.no_grad():
            out = self._model(x)                 # (1, window, n_features)
            per_feature_err = ((out - x) ** 2).mean(dim=1).squeeze(0).numpy()  # (n_features,)
            total_err       = float(per_feature_err.mean())

        # Top 3 most anomalous features
        top_idx     = np.argsort(per_feature_err)[::-1][:3]
        top_features = [(self.feature_cols[i], float(per_feature_err[i])) for i in top_idx]

        return AnomalyResult(
            is_anomaly           = total_err > self.threshold,
            reconstruction_error = total_err,
            threshold            = self.threshold,
            top_features         = top_features,
        )
