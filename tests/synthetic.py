"""Synthetic tennis-data and Sackmann files for the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenis.elo import win_prob

LAST = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliet",
        "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango"]


def simulate_tennis(n_days: int = 700, per_day: int = 6, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (tennis-data frame, Sackmann frame) generated from fixed player strengths."""
    rng = np.random.default_rng(seed)
    n = len(LAST)
    strength = rng.normal(1500, 120, n)
    clay_bonus = rng.normal(0, 60, n)
    serve = 0.62 + (strength - 1500) / 3000
    rows, sack = [], []
    start = pd.Timestamp("2020-01-06")
    surfaces = ["Hard", "Clay", "Grass"]
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        week = d // 7
        surface = surfaces[(week // 6) % 3]
        slam = week % 10 == 0
        for _ in range(per_day):
            i, j = rng.choice(n, 2, replace=False)
            ri = strength[i] + (clay_bonus[i] if surface == "Clay" else 0)
            rj = strength[j] + (clay_bonus[j] if surface == "Clay" else 0)
            p = float(win_prob(ri, rj))
            w, lo = (i, j) if rng.random() < p else (j, i)
            pw = p if w == i else 1 - p
            margin = 1.06
            noise_i, noise_j = np.exp(rng.normal(0, 0.04, 2))  # noise per player, independent of the result
            odds_i, odds_j = 1 / (p * margin) * noise_i, 1 / ((1 - p) * margin) * noise_j
            odds_w, odds_l = (odds_i, odds_j) if w == i else (odds_j, odds_i)
            comment = "Completed" if rng.random() > 0.03 else "Retired"
            rows.append({"ATP": week, "Location": "X", "Tournament": f"Event {week}", "Date": date,
                         "Series": "Grand Slam" if slam else "ATP250", "Court": "Outdoor", "Surface": surface,
                         "Round": "1st Round", "Best of": 5 if slam else 3,
                         "Winner": f"{LAST[w]} J.", "Loser": f"{LAST[lo]} J.",
                         "WRank": int(np.argsort(-strength).tolist().index(w)) + 1,
                         "LRank": int(np.argsort(-strength).tolist().index(lo)) + 1,
                         "Comment": comment, "B365W": odds_w * 0.99, "B365L": odds_l * 0.99,
                         "PSW": 1 / (pw * 1.02), "PSL": 1 / ((1 - pw) * 1.02),
                         "AvgW": odds_w, "AvgL": odds_l})
            if comment == "Completed":
                w_svpt, l_svpt = int(rng.integers(60, 100)), int(rng.integers(60, 100))
                w_won = int(rng.binomial(w_svpt, min(serve[w] + 0.03, 0.9)))
                l_won = int(rng.binomial(l_svpt, max(serve[lo] - 0.03, 0.3)))
                sack.append({"tourney_date": int((date - pd.Timedelta(days=date.weekday())).strftime("%Y%m%d")),
                             "tourney_name": f"Event {week}", "surface": surface, "best_of": 5 if slam else 3,
                             "winner_name": f"John {LAST[w]}", "loser_name": f"John {LAST[lo]}",
                             "score": "6-4 6-4", "w_svpt": w_svpt, "w_1stIn": 0, "w_1stWon": w_won, "w_2ndWon": 0,
                             "l_svpt": l_svpt, "l_1stIn": 0, "l_1stWon": l_won, "l_2ndWon": 0})
    return pd.DataFrame(rows), pd.DataFrame(sack)
