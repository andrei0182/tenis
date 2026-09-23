"""Expected value, fractional Kelly staking and bankroll simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StakingConfig:
    """Betting rules: minimum EV, Kelly fraction and per-bet cap (share of bankroll)."""

    ev_min: float = 0.03
    kelly: float = 0.25
    cap: float = 0.02
    bankroll: float = 1000.0


def expected_value(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """EV = p * odds - 1."""
    return np.asarray(p, float) * np.asarray(odds, float) - 1.0


def kelly_fraction(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Full Kelly f* = (p*odds - 1) / (odds - 1), floored at 0."""
    p, odds = np.asarray(p, float), np.asarray(odds, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * odds - 1.0) / (odds - 1.0)
    return np.clip(np.nan_to_num(f, nan=0.0), 0.0, None)


def stake_size(p: np.ndarray, odds: np.ndarray, bankroll: float, cfg: StakingConfig) -> np.ndarray:
    """min(kelly * f* * bankroll, cap * bankroll)."""
    return np.minimum(cfg.kelly * kelly_fraction(p, odds) * bankroll, cfg.cap * bankroll)


def best_outcome(p: np.ndarray, odds: np.ndarray, ev_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row: index of the max-EV outcome and its EV; index -1 when EV < ev_min."""
    ev = expected_value(p, odds)
    ev = np.where(np.isfinite(ev), ev, -np.inf)
    idx = np.argmax(ev, axis=1)
    best = ev[np.arange(len(ev)), idx]
    return np.where(best >= ev_min, idx, -1), best


def simulate_bankroll(bets: pd.DataFrame, cfg: StakingConfig) -> pd.DataFrame:
    """Settle bets day by day.

    `bets` needs columns date, p, odds, won. Stakes for one day are sized on
    the bankroll at the start of that day (bets placed before any settles) and
    scaled down if they would exceed it.
    """
    bets = bets.sort_values("date", kind="stable").reset_index(drop=True)
    bankroll = cfg.bankroll
    stakes, profits, after = [], [], []
    for _, day in bets.groupby("date", sort=True):
        s = stake_size(day["p"].to_numpy(), day["odds"].to_numpy(), bankroll, cfg)
        if s.sum() > bankroll:
            s *= bankroll / s.sum()
        won = day["won"].to_numpy(dtype=bool)
        pnl = np.where(won, s * (day["odds"].to_numpy() - 1.0), -s)
        stakes.append(s)
        profits.append(pnl)
        bankroll += pnl.sum()
        after.append(np.full(len(day), bankroll))
    if stakes:
        bets["stake"] = np.concatenate(stakes)
        bets["profit"] = np.concatenate(profits)
        bets["bankroll_after_day"] = np.concatenate(after)
    else:
        bets["stake"] = bets["profit"] = bets["bankroll_after_day"] = pd.Series(dtype=float)
    return bets[bets["stake"] > 0].reset_index(drop=True)
