"""Probabilistic scoring metrics for binary outcomes (1 = home win)."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def _arrays(y_true, p_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be 0/1")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be in [0, 1] and not NaN")
    return y, p


def log_loss(y_true, p_pred) -> float:
    """Mean binary cross-entropy (natural log). 0.693 = coin flip."""
    y, p = _arrays(y_true, p_pred)
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true, p_pred) -> float:
    """Mean squared error of probabilities. 0.25 = coin flip."""
    y, p = _arrays(y_true, p_pred)
    return float(np.mean((p - y) ** 2))


def accuracy(y_true, p_pred) -> float:
    """Share of games where the side given > 50% won (ties at 0.5 count as home)."""
    y, p = _arrays(y_true, p_pred)
    return float(np.mean((p >= 0.5) == (y == 1)))
