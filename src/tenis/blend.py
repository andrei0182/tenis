"""Odds de-vigging and logistic blend of model and market probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def devig_proportional(odds: np.ndarray) -> np.ndarray:
    """p_i = (1/o_i) / sum_j (1/o_j), row-wise. Rows with missing odds give NaN."""
    q = 1.0 / np.asarray(odds, dtype=float)
    return q / q.sum(axis=-1, keepdims=True)


def _power_row(q: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(q)) or np.any(q <= 0) or np.any(q >= 1):
        return np.full_like(q, np.nan)
    if abs(q.sum() - 1.0) < 1e-12:
        return q.copy()
    k = brentq(lambda k: np.sum(q ** k) - 1.0, 1e-3, 100.0)
    return q ** k


def devig_power(odds: np.ndarray) -> np.ndarray:
    """p_i = q_i^k with k chosen so the row sums to 1 (power method)."""
    q = 1.0 / np.atleast_2d(np.asarray(odds, dtype=float))
    if len(q) == 0:
        return q.reshape(np.shape(odds))
    out = np.vstack([_power_row(row) for row in q])
    return out.reshape(np.shape(odds))


def devig(odds: np.ndarray, method: str = "proportional") -> np.ndarray:
    """De-vig with the named method ('proportional' or 'power')."""
    if method == "power":
        return devig_power(odds)
    if method == "proportional":
        return devig_proportional(odds)
    raise ValueError(f"unknown de-vig method: {method}")


def logit(p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)) with clipping away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class MarketBlender:
    """Logistic regression on [logit(P_model), logit(P_market)] -> P_final.

    Binary markets use outcome 0 only. Markets with k > 2 outcomes are
    stacked one-vs-rest (plus outcome dummies) and renormalised.
    """

    C: float = 1.0
    min_rows: int = 200
    model: LogisticRegression | None = field(default=None, repr=False)
    n_train: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def _features(self, p_model: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        n, k = p_model.shape
        if k == 2:
            return np.column_stack([logit(p_model[:, 0]), logit(p_market[:, 0])])
        dummies = np.tile(np.eye(k)[:, 1:], (n, 1))
        return np.column_stack([logit(p_model.ravel()), logit(p_market.ravel()), dummies])

    def fit(self, p_model: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> "MarketBlender":
        """Fit on finished matches; y is the index of the winning outcome."""
        p_model, p_market, y = np.asarray(p_model, float), np.asarray(p_market, float), np.asarray(y)
        ok = np.isfinite(p_model).all(axis=1) & np.isfinite(p_market).all(axis=1)
        p_model, p_market, y = p_model[ok], p_market[ok], y[ok]
        self.n_train = len(y)
        if self.n_train < self.min_rows:
            self.model = None
            return self
        k = p_model.shape[1]
        target = (y == 0).astype(int) if k == 2 else (y[:, None] == np.arange(k)).astype(int).ravel()
        if len(np.unique(target)) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(C=self.C).fit(self._features(p_model, p_market), target)
        return self

    def predict(self, p_model: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        """Blended probabilities (n, k); NaN where inputs are missing or not fitted."""
        p_model, p_market = np.asarray(p_model, float), np.asarray(p_market, float)
        n, k = p_model.shape
        out = np.full((n, k), np.nan)
        ok = np.isfinite(p_model).all(axis=1) & np.isfinite(p_market).all(axis=1)
        if self.model is None or not ok.any():
            return out
        raw = self.model.predict_proba(self._features(p_model[ok], p_market[ok]))[:, 1]
        if k == 2:
            out[ok] = np.column_stack([raw, 1 - raw])
        else:
            raw = raw.reshape(-1, k)
            out[ok] = raw / raw.sum(axis=1, keepdims=True)
        return out

    @property
    def coefficients(self) -> dict[str, float]:
        """Weights on logit(model), logit(market) and the intercept."""
        if self.model is None:
            return {}
        c = self.model.coef_[0]
        return {"w_model": float(c[0]), "w_market": float(c[1]), "intercept": float(self.model.intercept_[0])}
