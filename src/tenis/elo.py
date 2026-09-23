"""General + surface Elo with experience-dependent K."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

START_RATING = 1500.0
W_GRID = tuple(np.round(np.linspace(0, 1, 11), 2))


def k_factor(n_matches: np.ndarray | int) -> np.ndarray | float:
    """K = 250 / (n + 5)^0.4 — decreases with the player's experience."""
    return 250.0 / (np.asarray(n_matches, dtype=float) + 5.0) ** 0.4


def win_prob(r_a: np.ndarray | float, r_b: np.ndarray | float) -> np.ndarray | float:
    """P(A beats B) = 1 / (1 + 10^((R_B - R_A) / 400))."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(r_b) - np.asarray(r_a)) / 400.0))


def _bo3(s: float) -> float:
    return s * s * (3.0 - 2.0 * s)


def set_prob_from_bo3(p: float) -> float:
    """Invert P(win best-of-3) = s^2 (3 - 2s) for the per-set probability s."""
    if p <= 0 or p >= 1:
        return float(np.clip(p, 0, 1))
    return brentq(lambda s: _bo3(s) - p, 0.0, 1.0)


def bo5_from_set(s: np.ndarray | float) -> np.ndarray | float:
    """P(win best-of-5) = s^3 (1 + 3q + 6q^2), q = 1 - s."""
    s = np.asarray(s, dtype=float)
    q = 1.0 - s
    return s ** 3 * (1.0 + 3.0 * q + 6.0 * q * q)


def to_best_of_5(p: np.ndarray | float) -> np.ndarray | float:
    """Convert a best-of-3 match probability to best-of-5 via the implied set probability."""
    arr = np.atleast_1d(np.asarray(p, dtype=float))
    out = np.array([bo5_from_set(set_prob_from_bo3(x)) if np.isfinite(x) else np.nan for x in arr])
    return out if np.ndim(p) else float(out[0])


@dataclass
class EloRatings:
    """Ratings state: one general rating and one per surface, plus match counts."""

    general: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: START_RATING))
    surface: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: START_RATING)))
    n_general: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_surface: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def ratings(self, player: str, surface: str) -> tuple[float, float]:
        """(general, surface) rating; unseen players start at 1500."""
        return self.general.get(player, START_RATING), self.surface.get(surface, {}).get(player, START_RATING)

    def update(self, winner: str, loser: str, surface: str) -> None:
        """R += K (result - P), separately for the general and the surface system."""
        g, s = self.general, self.surface[surface]
        ng, ns = self.n_general, self.n_surface[surface]
        p = win_prob(g[winner], g[loser])
        dw, dl = k_factor(ng[winner]) * (1 - p), k_factor(ng[loser]) * (1 - p)
        g[winner] += dw
        g[loser] -= dl
        p = win_prob(s[winner], s[loser])
        dw, dl = k_factor(ns[winner]) * (1 - p), k_factor(ns[loser]) * (1 - p)
        s[winner] += dw
        s[loser] -= dl
        ng[winner] += 1
        ng[loser] += 1
        ns[winner] += 1
        ns[loser] += 1

    def to_plain(self) -> dict:
        """Picklable snapshot (no lambdas)."""
        return {"general": dict(self.general), "surface": {k: dict(v) for k, v in self.surface.items()},
                "n_general": dict(self.n_general), "n_surface": {k: dict(v) for k, v in self.n_surface.items()}}


def run_elo(df: pd.DataFrame, ratings: EloRatings | None = None) -> tuple[pd.DataFrame, EloRatings]:
    """Pre-match ratings for every row (chronological); only completed matches update ratings."""
    ratings = ratings or EloRatings()
    cols = {k: np.empty(len(df)) for k in ("rg_w", "rg_l", "rs_w", "rs_l", "n_w", "n_l")}
    rows = zip(df["winner"], df["loser"], df["surface"], df["completed"])
    for i, (w, lo, surf, done) in enumerate(rows):
        cols["rg_w"][i], cols["rs_w"][i] = ratings.ratings(w, surf)
        cols["rg_l"][i], cols["rs_l"][i] = ratings.ratings(lo, surf)
        cols["n_w"][i], cols["n_l"][i] = ratings.n_general.get(w, 0), ratings.n_general.get(lo, 0)
        if done:
            ratings.update(w, lo, surf)
    out = df.copy()
    for k, v in cols.items():
        out[k] = v
    return out, ratings


def combined_prob(rg_a, rg_b, rs_a, rs_b, w: float) -> np.ndarray:
    """P(A wins) from R = w * R_surface + (1 - w) * R_general."""
    ra = w * np.asarray(rs_a) + (1 - w) * np.asarray(rg_a)
    rb = w * np.asarray(rs_b) + (1 - w) * np.asarray(rg_b)
    return win_prob(ra, rb)


def select_w(rg_a, rg_b, rs_a, rs_b, y, grid: tuple[float, ...] = W_GRID) -> tuple[float, dict[float, float]]:
    """Surface weight with the lowest log loss on pre-match (out-of-sample) ratings."""
    y = np.asarray(y)
    if len(y) == 0:
        return 0.5, {}
    losses = {}
    for w in grid:
        p = np.clip(combined_prob(rg_a, rg_b, rs_a, rs_b, w), 1e-12, 1 - 1e-12)
        losses[float(w)] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return min(losses, key=losses.get), losses


@dataclass
class EloModel:
    """Fitted Elo state for `predict`."""

    ratings: dict
    w: float

    def predict(self, player_a: str, player_b: str, surface: str, best_of: int = 3) -> float:
        """P(A beats B); best_of=5 converts via the per-set probability."""
        g, s = self.ratings["general"], self.ratings["surface"].get(surface, {})
        p = float(combined_prob(g.get(player_a, START_RATING), g.get(player_b, START_RATING),
                                s.get(player_a, START_RATING), s.get(player_b, START_RATING), self.w))
        return float(to_best_of_5(p)) if best_of == 5 else p
