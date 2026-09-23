"""Command line: tenis fit / predict / backtest."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from .backtest import FittedTennis, TennisConfig, fit_system, run_backtest
from .blend import devig
from .data import normalize_player, normalize_surface
from .markov import match_distribution
from .staking import StakingConfig, expected_value, stake_size

app = typer.Typer(help="Surface Elo + Markov tennis model, market blend, Kelly, walk-forward backtest.",
                  no_args_is_help=True)


@app.command()
def fit(
    data: Path = typer.Option(Path("data/raw"), help="Folder with tennis-data .xlsx/.csv files."),
    out: Path = typer.Option(Path("outputs/elo.pkl")),
    sackmann: Path | None = typer.Option(None, help="Sackmann CSV folder (default: <data>/sackmann)."),
    player_map: Path | None = typer.Option(None, help="CSV sackmann_name,td_name (default: data/player_map.csv)."),
    w: float | None = typer.Option(None, help="Fixed surface weight; default: chosen on the data."),
    odds_source: str = typer.Option("Avg", help="Bet-time odds: Avg or B365."),
    seed: int = typer.Option(42),
) -> None:
    """Run the ratings over all completed matches and train the market blenders."""
    cfg = TennisConfig(start=pd.Timestamp("2100-01-01"), w=w, seed=seed)
    system = fit_system(data, cfg, odds_source, sackmann, player_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(system))
    typer.echo(f"Jucători cu rating: {len(system.elo.ratings['general'])}, w (suprafață) = {system.elo.w}")
    typer.echo(f"Markov: {'da' if system.serve_model else 'nu (lipsesc datele Sackmann)'}")
    typer.echo(f"Model salvat în {out}")


@app.command()
def predict(
    model: Path = typer.Option(Path("outputs/elo.pkl")),
    player_a: str = typer.Option(..., help='Name as in tennis-data, e.g. "Sinner J."'),
    player_b: str = typer.Option(...),
    surface: str = typer.Option("Hard", help="Hard, Clay or Grass."),
    best_of_5: bool = typer.Option(False, "--best-of-5", help="Grand Slam men's match (best of 5)."),
    odds_a: float | None = typer.Option(None), odds_b: float | None = typer.Option(None),
    ev_min: float = typer.Option(0.03), kelly: float = typer.Option(0.25), cap: float = typer.Option(0.02),
    bankroll: float = typer.Option(1000.0),
) -> None:
    """P(A beats B) from Elo (and Markov when fitted), blended with the market if odds are given."""
    system: FittedTennis = pickle.loads(model.read_bytes())
    a, b, surf, best_of = normalize_player(player_a), normalize_player(player_b), normalize_surface(surface), \
        (5 if best_of_5 else 3)
    for key, name in ((a, player_a), (b, player_b)):
        if key not in system.elo.ratings["general"]:
            typer.echo(f"Atenție: {name} nu are istoric — rating inițial 1500.")
    probs = {"elo": system.elo.predict(a, b, surf, best_of)}
    if system.serve_model is not None:
        pa, pb = system.serve_model.serve_probs(a, b, surf)
        if np.isfinite(pa):
            dist = match_distribution(pa, pb, best_of)
            probs["markov"] = dist.p_match
            typer.echo(f"Markov: p_serviciu A={pa:.3f}, B={pb:.3f}; seturi {dist.set_scores}")
    for name, p in probs.items():
        typer.echo(f"P({player_a} câștigă) — {name}: {p:.3f}")
    if odds_a and odds_b:
        q = float(devig(np.array([[odds_a, odds_b]]), system.devig_method)[0, 0])
        typer.echo(f"Piață (de-vig): {q:.3f}")
        staking = StakingConfig(ev_min=ev_min, kelly=kelly, cap=cap, bankroll=bankroll)
        for name, p in probs.items():
            f = system.blenders[name].predict(np.array([[p, 1 - p]]), np.array([[q, 1 - q]]))[0, 0]
            if not np.isfinite(f):
                typer.echo(f"blend_{name}: blender neantrenat (prea puține meciuri cu cote)")
                continue
            for who, pp, o in ((player_a, f, odds_a), (player_b, 1 - f, odds_b)):
                ev = float(expected_value(pp, o))
                bet = ev >= ev_min
                stake = float(stake_size(pp, o, bankroll, staking)) if bet else 0.0
                typer.echo(f"blend_{name}: {who} P={pp:.3f} cotă={o} EV={ev:+.3f}"
                           + (f" -> PARIU, miză {stake:.2f}" if bet else ""))


@app.command()
def backtest(
    data: Path = typer.Option(Path("data/raw")),
    start: str = typer.Option(..., help="First date with bets (YYYY-MM-DD)."),
    surface: str | None = typer.Option(None, help="Filter: hard / clay / grass."),
    series: str | None = typer.Option(None, help='Filter: e.g. "Grand Slam", "Masters", "ATP 250".'),
    model: str = typer.Option("elo", help="Probability blended with the market for bets: elo or markov."),
    ev_min: float = typer.Option(0.03), kelly: float = typer.Option(0.25), cap: float = typer.Option(0.02),
    bankroll: float = typer.Option(1000.0), refit_days: int = typer.Option(30),
    w: float | None = typer.Option(None), odds_source: str = typer.Option("Avg"),
    devig_method: str = typer.Option("proportional"),
    sackmann: Path | None = typer.Option(None), player_map: Path | None = typer.Option(None),
    seed: int = typer.Option(42), out_dir: Path = typer.Option(Path("outputs")),
) -> None:
    """Walk-forward backtest with Elo vs Markov vs blend comparison."""
    cfg = TennisConfig(start=pd.Timestamp(start), staking=StakingConfig(ev_min=ev_min, kelly=kelly, cap=cap,
                       bankroll=bankroll), refit_days=refit_days, w=w, bet_model=model, surface=surface,
                       series=series, seed=seed, devig_method=devig_method)
    summary = run_backtest(data, cfg, out_dir, odds_source, sackmann, player_map)
    typer.echo(format_summary(summary))
    typer.echo(f"\nRezultate în {out_dir}/ (bets.csv, comparison.csv, predictions.csv, summary.json)")


def _pct(v: float | None) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.2f}%"


def format_summary(summary: dict) -> str:
    """Human-readable report."""
    b = summary["betting"]
    lines = [
        f"Meciuri evaluate (>= start, după filtre): {summary['data']['evaluated']}, "
        f"cu Markov: {summary['data']['markov_available']}, w suprafață = {summary['elo_surface_weight_w']}",
        f"Strategie {summary['strategy']}: {b['n_bets']} pariuri | hit rate {_pct(b['hit_rate'])} | "
        f"yield {_pct(b['yield'])} | ROI bankroll {_pct(b['roi_bankroll'])} | "
        f"max drawdown {_pct(b['max_drawdown'])} | CLV mediu {_pct(b['clv_mean'])} (n={b['clv_n']})",
        "", "Comparație (walk-forward):",
    ]
    table = pd.DataFrame(summary["comparison"])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        lines.append(table.round(4).to_string(index=False))
    lines += [""] + summary["verdicts"]
    for title, key in (("Pe suprafață", "by_surface"), ("Pe nivel turneu", "by_level")):
        lines.append(f"\n{title}:")
        for group, e in summary[key].items():
            ll = f"model {e['model']['log_loss']:.4f} vs piață {e['market']['log_loss']:.4f}" if "model" in e else "-"
            lines.append(f"  {group}: n={e['n_matches']}, log loss {ll}, pariuri {e['betting']['n_bets']}, "
                         f"yield {_pct(e['betting']['yield'])}, CLV {_pct(e['betting']['clv_mean'])}")
    if summary["sackmann"] and summary["sackmann"]["unmatched_count"]:
        lines.append(f"\nSackmann: {summary['sackmann']['unmatched_count']} nume nepotrivite "
                     "(vezi outputs/unmatched_players.csv; completează data/player_map.csv)")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    app()
