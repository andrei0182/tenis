"""Walk-forward backtest for tennis: Elo vs Markov vs blend with the market."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .blend import MarketBlender, devig
from .data import build_player_map, load_sackmann, load_tennis, to_ab
from .elo import EloModel, combined_prob, run_elo, select_w, to_best_of_5
from .markov import ServeReturnModel, markov_predictions
from .staking import StakingConfig, best_outcome, simulate_bankroll

EPS = 1e-12
LEVEL_ORDER = ("Grand Slam", "Masters", "ATP 500", "ATP 250")


@dataclass
class TennisConfig:
    """Walk-forward settings; filters only restrict evaluation and bets, never training."""

    start: pd.Timestamp
    staking: StakingConfig = field(default_factory=StakingConfig)
    refit_days: int = 30
    blend_min: int = 300
    w: float | None = None
    burn_in_days: int = 365
    bo5_adjust: bool = True
    bet_model: str = "elo"
    surface: str | None = None
    series: str | None = None
    seed: int = 42
    devig_method: str = "proportional"
    sackmann_lag_days: int = 15


def _surface_filter(value: str | None) -> str | None:
    return value.strip().capitalize() if value else None


def elo_probabilities(ab: pd.DataFrame, w: float, bo5_adjust: bool) -> np.ndarray:
    """P(A wins) from pre-match ratings; best-of-5 matches converted via the set probability."""
    a_won = ab["y"].to_numpy() == 1
    rg_a = np.where(a_won, ab["rg_w"], ab["rg_l"])
    rg_b = np.where(a_won, ab["rg_l"], ab["rg_w"])
    rs_a = np.where(a_won, ab["rs_w"], ab["rs_l"])
    rs_b = np.where(a_won, ab["rs_l"], ab["rs_w"])
    p = combined_prob(rg_a, rg_b, rs_a, rs_b, w)
    if bo5_adjust:
        bo5 = ab["best_of"].to_numpy() == 5
        if bo5.any():
            p = p.copy()
            p[bo5] = to_best_of_5(p[bo5])
    return p


def _ratings_ab(ab: pd.DataFrame) -> tuple[np.ndarray, ...]:
    a_won = ab["y"].to_numpy() == 1
    return (np.where(a_won, ab["rg_w"], ab["rg_l"]), np.where(a_won, ab["rg_l"], ab["rg_w"]),
            np.where(a_won, ab["rs_w"], ab["rs_l"]), np.where(a_won, ab["rs_l"], ab["rs_w"]))


def choose_w(ab: pd.DataFrame, cfg: TennisConfig) -> tuple[float, dict]:
    """Surface weight chosen on matches before `start` (after the burn-in), pre-match ratings only."""
    if cfg.w is not None:
        return cfg.w, {}
    first = ab["date"].min() + pd.Timedelta(days=cfg.burn_in_days)
    mask = ((ab["date"] >= first) & (ab["date"] < cfg.start) & (ab["best_of"] == 3)).to_numpy()
    rg_a, rg_b, rs_a, rs_b = _ratings_ab(ab)
    return select_w(rg_a[mask], rg_b[mask], rs_a[mask], rs_b[mask], ab["y"].to_numpy()[mask])


def walk_forward_blend(ab: pd.DataFrame, model_col: str, cfg: TennisConfig) -> np.ndarray:
    """P_final for A: each block's blender is fit on matches finished before the block."""
    out = np.full(len(ab), np.nan)
    p_model = np.column_stack([ab[model_col], 1 - ab[model_col]])
    p_market = np.column_stack([ab["q_a"], 1 - ab["q_a"]])
    y = np.where(ab["y"].to_numpy() == 1, 0, 1)  # outcome 0 = A wins
    blender = MarketBlender(min_rows=cfg.blend_min)
    dates = ab["date"]
    for t in pd.date_range(dates.min(), dates.max(), freq=f"{cfg.refit_days}D"):
        block = ((dates >= t) & (dates < t + pd.Timedelta(days=cfg.refit_days))).to_numpy()
        if not block.any():
            continue
        past = (dates < t).to_numpy()
        blender.fit(p_model[past], p_market[past], y[past])
        out[block] = blender.predict(p_model[block], p_market[block])[:, 0]
    return out


def prob_scores(p: np.ndarray, y: np.ndarray) -> dict:
    """Binary log loss and Brier score."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


def candidate_bets(ab: pd.DataFrame, prob_col: str, cfg: TennisConfig) -> pd.DataFrame:
    """Back whichever player has the larger EV, if EV >= ev_min."""
    p_a = ab[prob_col].to_numpy(dtype=float)
    p = np.column_stack([p_a, 1 - p_a])
    odds = ab[["odds_a", "odds_b"]].to_numpy(dtype=float)
    close = ab[["close_a", "close_b"]].to_numpy(dtype=float)
    idx, ev = best_outcome(p, odds, cfg.staking.ev_min)
    take = idx >= 0
    rows, idx, ev = ab[take], idx[take], ev[take]
    r = np.arange(len(rows))
    q = np.column_stack([rows["q_a"], 1 - rows["q_a"]])
    return pd.DataFrame({
        "date": rows["date"].to_numpy(), "tournament": rows["tournament"].to_numpy(),
        "surface": rows["surface"].to_numpy(), "level": rows["level"].to_numpy(),
        "round": rows["round"].to_numpy(),
        "player": np.where(idx == 0, rows["name_a"], rows["name_b"]),
        "opponent": np.where(idx == 0, rows["name_b"], rows["name_a"]),
        "p": p[take][r, idx], "p_market": q[r, idx], "odds": odds[take][r, idx],
        "close_odds": close[take][r, idx], "ev": ev,
        "won": np.where(idx == 0, rows["y"] == 1, rows["y"] == 0),
    })


def max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough fall as a share of the peak."""
    if len(curve) == 0:
        return 0.0
    peaks = np.maximum.accumulate(curve)
    return float(np.max((peaks - curve) / peaks))


def betting_metrics(bets: pd.DataFrame, bankroll0: float) -> dict:
    """Bets, hit rate, yield, ROI on bankroll, max drawdown and CLV vs Pinnacle."""
    if bets.empty:
        return {"n_bets": 0, "hit_rate": None, "staked": 0.0, "profit": 0.0, "yield": None,
                "roi_bankroll": 0.0, "final_bankroll": bankroll0, "max_drawdown": 0.0,
                "clv_mean": None, "clv_n": 0, "clv_positive_share": None}
    daily = bets.groupby("date")["bankroll_after_day"].last().to_numpy()
    clv = (bets["odds"] / bets["close_odds"] - 1).dropna()
    staked, profit = float(bets["stake"].sum()), float(bets["profit"].sum())
    return {
        "n_bets": int(len(bets)), "hit_rate": float(bets["won"].mean()), "staked": staked,
        "profit": profit, "yield": profit / staked if staked else None,
        "roi_bankroll": profit / bankroll0, "final_bankroll": bankroll0 + profit,
        "max_drawdown": max_drawdown(np.concatenate([[bankroll0], daily])),
        "clv_mean": float(clv.mean()) if len(clv) else None, "clv_n": int(len(clv)),
        "clv_positive_share": float((clv > 0).mean()) if len(clv) else None,
    }


def prepare(data_dir: str | Path, cfg: TennisConfig, odds_source: str = "Avg",
            sackmann_dir: str | Path | None = None, map_csv: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load data, run Elo (+ Markov), orient A/B, add market and blended probabilities."""
    data_dir = Path(data_dir)
    raw = load_tennis(data_dir, odds_source)
    rated, ratings = run_elo(raw)
    ab = to_ab(rated[rated["completed"]].reset_index(drop=True), cfg.seed)
    w, w_losses = choose_w(ab, cfg)
    ab["p_elo"] = elo_probabilities(ab, w, cfg.bo5_adjust)

    info: dict = {"w": w, "w_losses": w_losses, "elo_state": ratings.to_plain(), "markov": None}
    sackmann_dir = Path(sackmann_dir) if sackmann_dir else data_dir / "sackmann"
    sack = load_sackmann(sackmann_dir)
    ab["p_markov"] = np.nan
    if sack is not None:
        td_keys = set(ab["player_a"]) | set(ab["player_b"])
        names = set(sack["winner_name"]) | set(sack["loser_name"])
        name_map, unmatched = build_player_map(td_keys, names, map_csv or data_dir.parent / "player_map.csv")
        p_markov, serve_model = markov_predictions(ab, sack, name_map, cfg.sackmann_lag_days)
        ab["p_markov"] = p_markov
        info["markov"] = {"mapped": len(name_map), "unmatched": unmatched, "state": serve_model}

    ab["q_a"] = devig(ab[["odds_a", "odds_b"]].to_numpy(dtype=float), cfg.devig_method)[:, 0]
    ab["pin_a"] = devig(ab[["close_a", "close_b"]].to_numpy(dtype=float), cfg.devig_method)[:, 0]
    ab["f_elo"] = walk_forward_blend(ab, "p_elo", cfg)
    ab["f_markov"] = walk_forward_blend(ab, "p_markov", cfg) if ab["p_markov"].notna().any() else np.nan
    info["blend_rows"] = int(ab["q_a"].notna().sum())
    return ab, info


def evaluation_rows(ab: pd.DataFrame, cfg: TennisConfig) -> pd.DataFrame:
    """Out-of-sample rows (date >= start) after the --surface / --series filters."""
    rows = ab[ab["date"] >= cfg.start]
    surface = _surface_filter(cfg.surface)
    if surface:
        rows = rows[rows["surface"] == surface]
    if cfg.series:
        s = cfg.series.strip().lower()
        rows = rows[(rows["series"].str.lower() == s) | (rows["level"].str.lower() == s)]
    return rows


STRATEGIES = {"elo": "p_elo", "markov": "p_markov", "blend_elo": "f_elo", "blend_markov": "f_markov"}


def comparison_table(rows: pd.DataFrame, cfg: TennisConfig) -> pd.DataFrame:
    """Log loss, Brier, yield, ROI and CLV per probability source, each vs the market on the same rows."""
    y = rows["y"].to_numpy()
    records = []
    for name, col in STRATEGIES.items():
        ok = (rows[col].notna() & rows["q_a"].notna()).to_numpy()
        if not ok.any():
            continue
        s, m = prob_scores(rows[col].to_numpy()[ok], y[ok]), prob_scores(rows["q_a"].to_numpy()[ok], y[ok])
        bets = simulate_bankroll(candidate_bets(rows[ok], col, cfg), cfg.staking)
        bm = betting_metrics(bets, cfg.staking.bankroll)
        records.append({"source": name, "n": int(ok.sum()), "log_loss": s["log_loss"], "brier": s["brier"],
                        "market_log_loss": m["log_loss"], "market_brier": m["brier"],
                        "beats_market": s["log_loss"] < m["log_loss"], "n_bets": bm["n_bets"],
                        "yield": bm["yield"], "roi_bankroll": bm["roi_bankroll"], "clv_mean": bm["clv_mean"]})
    for name, col in (("market_bet_time", "q_a"), ("pinnacle_close", "pin_a")):
        ok = rows[col].notna().to_numpy()
        if ok.any():
            s = prob_scores(rows[col].to_numpy()[ok], y[ok])
            records.append({"source": name, "n": int(ok.sum()), "log_loss": s["log_loss"], "brier": s["brier"]})
    return pd.DataFrame(records)


def group_report(rows: pd.DataFrame, bets: pd.DataFrame, prob_col: str, by: str, cfg: TennisConfig) -> dict:
    """Probability quality and betting results per surface or tournament level."""
    out = {}
    for key, grp in rows.groupby(by):
        ok = (grp[prob_col].notna() & grp["q_a"].notna()).to_numpy()
        entry: dict = {"n_matches": int(len(grp))}
        if ok.any():
            y = grp["y"].to_numpy()[ok]
            entry["model"] = prob_scores(grp[prob_col].to_numpy()[ok], y)
            entry["market"] = prob_scores(grp["q_a"].to_numpy()[ok], y)
            entry["model_beats_market"] = entry["model"]["log_loss"] < entry["market"]["log_loss"]
        entry["betting"] = betting_metrics(bets[bets[by] == key], cfg.staking.bankroll)
        out[str(key)] = entry
    return out


def verdict(table: pd.DataFrame, source: str) -> str:
    """Plain statement of whether `source` beats the bet-time market on log loss."""
    row = table[table["source"] == source]
    if row.empty:
        return f"{source}: fără date comparabile cu piața."
    if bool(row["beats_market"].iloc[0]):
        return f"{source}: bate piața la log loss walk-forward."
    return (f"ATENȚIE: {source} NU bate piața la log loss walk-forward -> nu are edge demonstrat, "
            "indiferent de ROI-ul pe perioada testată.")


def run_backtest(data_dir: str | Path, cfg: TennisConfig, out_dir: str | Path = "outputs",
                 odds_source: str = "Avg", sackmann_dir: str | Path | None = None,
                 map_csv: str | Path | None = None) -> dict:
    """Full pipeline with exports: bets.csv, predictions.csv, comparison.csv, summary.json."""
    ab, info = prepare(data_dir, cfg, odds_source, sackmann_dir, map_csv)
    rows = evaluation_rows(ab, cfg)
    main_col = STRATEGIES[f"blend_{cfg.bet_model}"]
    if rows[main_col].isna().all():
        raise ValueError(f"no blended probabilities for '{cfg.bet_model}' after {cfg.start.date()}")
    bets = simulate_bankroll(candidate_bets(rows[rows[main_col].notna()], main_col, cfg), cfg.staking)
    table = comparison_table(rows, cfg)

    summary = {
        "config": {**{k: v for k, v in asdict(cfg).items() if k not in ("start", "staking")},
                   "start": str(cfg.start.date()), "staking": asdict(cfg.staking), "odds_source": odds_source},
        "data": {"matches_completed": int(len(ab)), "evaluated": int(len(rows)),
                 "markov_available": int(rows["p_markov"].notna().sum())},
        "elo_surface_weight_w": info["w"],
        "sackmann": None if info["markov"] is None else
        {"mapped_players": info["markov"]["mapped"], "unmatched_count": len(info["markov"]["unmatched"]),
         "unmatched_sample": info["markov"]["unmatched"][:50]},
        "strategy": f"blend_{cfg.bet_model}",
        "betting": betting_metrics(bets, cfg.staking.bankroll),
        "comparison": table.to_dict(orient="records"),
        "verdicts": [verdict(table, s) for s in ("elo", "markov", "blend_elo", "blend_markov")
                     if s in set(table["source"])],
        "by_surface": group_report(rows, bets, main_col, "surface", cfg),
        "by_level": group_report(rows, bets, main_col, "level", cfg),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bets.to_csv(out_dir / "bets.csv", index=False)
    table.to_csv(out_dir / "comparison.csv", index=False)
    keep = ["date", "tournament", "series", "level", "surface", "round", "best_of", "name_a", "name_b", "y",
            "odds_a", "odds_b", "close_a", "close_b", "q_a", "pin_a", "p_elo", "p_markov", "f_elo", "f_markov"]
    rows[keep].to_csv(out_dir / "predictions.csv", index=False)
    if info["markov"] is not None and info["markov"]["unmatched"]:
        pd.DataFrame({"sackmann_name": info["markov"]["unmatched"], "td_name": ""}).to_csv(
            out_dir / "unmatched_players.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


@dataclass
class FittedTennis:
    """What `predict` needs: Elo state, surface weight, optional serve model and blenders."""

    elo: EloModel
    serve_model: ServeReturnModel | None
    blenders: dict[str, MarketBlender]
    devig_method: str
    fitted_until: pd.Timestamp


def fit_system(data_dir: str | Path, cfg: TennisConfig, odds_source: str = "Avg",
               sackmann_dir: str | Path | None = None, map_csv: str | Path | None = None) -> FittedTennis:
    """Ratings on all completed matches; w chosen on all data; blenders on out-of-sample probabilities."""
    ab, info = prepare(data_dir, cfg, odds_source, sackmann_dir, map_csv)
    blenders = {}
    y = np.where(ab["y"].to_numpy() == 1, 0, 1)
    for name in ("elo", "markov"):
        col = STRATEGIES[name]
        p = np.column_stack([ab[col], 1 - ab[col]])
        q = np.column_stack([ab["q_a"], 1 - ab["q_a"]])
        blenders[name] = MarketBlender(min_rows=cfg.blend_min).fit(p, q, y)
    serve = info["markov"]["state"] if info["markov"] else None
    return FittedTennis(EloModel(info["elo_state"], info["w"]), serve, blenders, cfg.devig_method,
                        ab["date"].max())
