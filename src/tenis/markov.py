"""Klaassen-Magnus style point-based Markov model: point -> game -> tiebreak -> set -> match."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

MAX_GAMES = 5 * 13


def p_game(p: float) -> float:
    """P(server holds) when winning each service point with probability p (exact, incl. deuce)."""
    q = 1.0 - p
    deuce = p * p / (1.0 - 2.0 * p * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


def p_tiebreak(pa: float, pb: float, a_serves_first: bool = True) -> float:
    """P(A wins a 7-point tiebreak); pa/pb = P(point won on own serve)."""
    x, y = pa, 1.0 - pb  # A wins a point on A's / on B's serve
    probs = defaultdict(float)
    probs[(0, 0)] = 1.0
    win = 0.0
    for total in range(12):  # all states with i + j = total, i, j <= 6
        for i in range(max(0, total - 6), min(total, 6) + 1):
            j = total - i
            mass = probs.pop((i, j), 0.0)
            if mass == 0.0:
                continue
            first_serves = ((total + 1) // 2) % 2 == 0
            a_serving = first_serves == a_serves_first
            pw = x if a_serving else y
            if i + 1 == 7:
                win += mass * pw
            else:
                probs[(i + 1, j)] += mass * pw
            if j + 1 < 7:
                probs[(i, j + 1)] += mass * (1 - pw)
    both, lose_both = x * y, (1 - x) * (1 - y)
    deuce = both / (both + lose_both) if both + lose_both > 0 else 0.5
    return win + probs.get((6, 6), 0.0) * deuce


@lru_cache(maxsize=4096)
def set_score_distribution(pa: float, pb: float, a_serves_first: bool) -> dict[tuple[int, int], float]:
    """Distribution of final set scores (games A, games B); 7-6 / 6-7 via tiebreak."""
    ha, hb = p_game(pa), p_game(pb)  # A holds / B holds
    states = {(0, 0): 1.0}
    final: dict[tuple[int, int], float] = defaultdict(float)
    while states:
        nxt: dict[tuple[int, int], float] = defaultdict(float)
        for (i, j), mass in states.items():
            if (i, j) == (6, 6):
                t = p_tiebreak(pa, pb, a_serves_first)
                final[(7, 6)] += mass * t
                final[(6, 7)] += mass * (1 - t)
                continue
            a_serving = ((i + j) % 2 == 0) == a_serves_first
            pw = ha if a_serving else 1.0 - hb
            for (ni, nj), pm in (((i + 1, j), pw), ((i, j + 1), 1 - pw)):
                done = (max(ni, nj) >= 6 and abs(ni - nj) >= 2) or max(ni, nj) == 7
                (final if done else nxt)[(ni, nj)] += mass * pm
        states = nxt
    return dict(final)


def p_set(pa: float, pb: float) -> float:
    """P(A wins a set), averaged over who serves first."""
    out = 0.0
    for first in (True, False):
        out += 0.5 * sum(v for (i, j), v in set_score_distribution(pa, pb, first).items() if i > j)
    return out


@dataclass
class MatchDistribution:
    """Match win probability plus derived markets."""

    p_match: float
    set_scores: dict[str, float]
    total_games: np.ndarray  # P(total games == g)

    def p_total_games_over(self, line: float) -> float:
        g = np.arange(len(self.total_games))
        return float(self.total_games[g > line].sum())

    def p_set_handicap(self, line: float) -> float:
        """P(sets_A - sets_B + line > 0), e.g. line=-1.5 means A wins 2-0 (Bo3)."""
        return float(sum(p for s, p in self.set_scores.items()
                         if int(s.split("-")[0]) - int(s.split("-")[1]) + line > 0))


def match_distribution(pa: float, pb: float, best_of: int = 3) -> MatchDistribution:
    """Exact set-by-set chain; the first server of each set follows from the previous set's game count.

    The first server of the match is unknown, so both cases are averaged.
    """
    need = best_of // 2 + 1
    pa, pb = round(float(pa), 6), round(float(pb), 6)
    states: dict[tuple[int, int, bool], np.ndarray] = {}
    for first in (True, False):
        vec = np.zeros(MAX_GAMES + 1)
        vec[0] = 0.5
        states[(0, 0, first)] = vec
    finished: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(MAX_GAMES + 1))
    while states:
        nxt: dict[tuple[int, int, bool], np.ndarray] = {}
        for (sa, sb, first), games in states.items():
            for (ga, gb), p in set_score_distribution(pa, pb, first).items():
                n = ga + gb
                shifted = np.zeros_like(games)
                shifted[n:] = games[:len(games) - n] * p
                na, nb = sa + (ga > gb), sb + (gb > ga)
                next_first = first if n % 2 == 0 else not first
                if na == need or nb == need:
                    finished[f"{na}-{nb}"] += shifted
                else:
                    key = (na, nb, next_first)
                    nxt[key] = nxt.get(key, 0) + shifted
        states = nxt
    set_scores = {k: float(v.sum()) for k, v in sorted(finished.items())}
    p_match = sum(p for k, p in set_scores.items() if int(k.split("-")[0]) == need)
    total = sum(finished.values())
    return MatchDistribution(p_match=float(p_match), set_scores=set_scores, total_games=total)


def p_match(pa: float, pb: float, best_of: int = 3) -> float:
    """P(A wins the match)."""
    return match_distribution(pa, pb, best_of).p_match


# --- serve / return strengths from past matches --------------------------------


@dataclass
class ServeReturnModel:
    """Time-decayed serve and return point percentages.

    p_A = f_t + (f_A - f_avg) - (g_B - g_avg), where f = serve points won %,
    g = return points won %, f_t = average serve % on the surface (proxy for
    the tournament), all computed only from matches already available.
    """

    xi: float = 0.002  # decay per day (half-life ~ 350 days)
    prior_points: float = 300.0  # shrinkage towards the tour average
    min_points: float = 200.0  # below this (decayed) serve points the estimate is NaN
    players: dict[str, np.ndarray] = field(default_factory=dict)  # [sp_won, sp_tot, rp_won, rp_tot]
    surfaces: dict[str, np.ndarray] = field(default_factory=dict)  # [sp_won, sp_tot]
    last: pd.Timestamp | None = None

    def _decay_to(self, date: pd.Timestamp) -> None:
        if self.last is not None and date > self.last:
            f = np.exp(-self.xi * (date - self.last).days)
            for arr in list(self.players.values()) + list(self.surfaces.values()):
                arr *= f
        self.last = date if self.last is None else max(self.last, date)

    def add(self, date: pd.Timestamp, surface: str, winner: str, loser: str,
            w_sp_won: float, w_sp: float, l_sp_won: float, l_sp: float) -> None:
        """Add one finished match (serve points won / played for each side)."""
        self._decay_to(date)
        w = self.players.setdefault(winner, np.zeros(4))
        lo = self.players.setdefault(loser, np.zeros(4))
        w += [w_sp_won, w_sp, l_sp - l_sp_won, l_sp]
        lo += [l_sp_won, l_sp, w_sp - w_sp_won, w_sp]
        self.surfaces.setdefault(surface, np.zeros(2))[:] += [w_sp_won + l_sp_won, w_sp + l_sp]

    def _tour_avg(self) -> float:
        tot = sum(self.surfaces.values(), np.zeros(2))
        return tot[0] / tot[1] if tot[1] > 0 else 0.62

    def serve_probs(self, a: str, b: str, surface: str, date: pd.Timestamp | None = None) -> tuple[float, float]:
        """(p_A, p_B): probability of winning a point on own serve; NaN if too little data."""
        if date is not None:
            self._decay_to(date)
        sa, sb = self.players.get(a), self.players.get(b)
        if sa is None or sb is None or sa[1] < self.min_points or sb[1] < self.min_points:
            return np.nan, np.nan
        f_avg = self._tour_avg()
        g_avg = 1.0 - f_avg
        surf = self.surfaces.get(surface)
        f_t = surf[0] / surf[1] if surf is not None and surf[1] > 0 else f_avg
        k = self.prior_points

        def f(s: np.ndarray) -> float:
            return (s[0] + k * f_avg) / (s[1] + k)

        def g(s: np.ndarray) -> float:
            return (s[2] + k * g_avg) / (s[3] + k)

        pa = f_t + (f(sa) - f_avg) - (g(sb) - g_avg)
        pb = f_t + (f(sb) - f_avg) - (g(sa) - g_avg)
        return float(np.clip(pa, 0.05, 0.95)), float(np.clip(pb, 0.05, 0.95))


def markov_predictions(matches: pd.DataFrame, sackmann: pd.DataFrame, name_map: dict[str, str],
                       lag_days: int = 15, model: ServeReturnModel | None = None) -> tuple[np.ndarray, ServeReturnModel]:
    """P(A wins) for each row of `matches` (columns date, player_a, player_b, surface, best_of).

    A Sackmann match only becomes usable `lag_days` after its tourney_date
    (Sackmann has no per-match dates), so no statistic from the same or a
    later tournament can leak into a prediction.
    """
    model = model or ServeReturnModel()
    stats = sackmann.assign(available=sackmann["tourney_date"] + pd.Timedelta(days=lag_days))
    stats = stats.sort_values("available", kind="stable")
    stats = stats[stats["winner_name"].isin(name_map) & stats["loser_name"].isin(name_map)]
    order = np.argsort(matches["date"].to_numpy(), kind="stable")
    out = np.full(len(matches), np.nan)
    rows = stats[["available", "surface", "winner_name", "loser_name", "w_sp_won", "w_svpt", "l_sp_won", "l_svpt"]]
    it = rows.itertuples(index=False)
    pending = next(it, None)
    cols = [matches[c].to_numpy() for c in ("date", "player_a", "player_b", "surface", "best_of")]
    for i in order:
        date, a, b, surface, best_of = (c[i] for c in cols)
        date = pd.Timestamp(date)
        while pending is not None and pending.available < date:
            model.add(pending.available, pending.surface, name_map[pending.winner_name],
                      name_map[pending.loser_name], pending.w_sp_won, pending.w_svpt,
                      pending.l_sp_won, pending.l_svpt)
            pending = next(it, None)
        pa, pb = model.serve_probs(a, b, surface, date)
        if np.isfinite(pa):
            out[i] = p_match(pa, pb, int(best_of))
    while pending is not None:  # keep the final state complete for `predict`
        model.add(pending.available, pending.surface, name_map[pending.winner_name],
                  name_map[pending.loser_name], pending.w_sp_won, pending.w_svpt,
                  pending.l_sp_won, pending.l_svpt)
        pending = next(it, None)
    return out, model
