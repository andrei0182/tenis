import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from synthetic import simulate_tennis
from tenis.backtest import TennisConfig, evaluation_rows, max_drawdown, prepare, run_backtest
from tenis.cli import app
from tenis.staking import StakingConfig


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    raw = root / "raw"
    (raw / "sackmann").mkdir(parents=True)
    td, sack = simulate_tennis(n_days=500, per_day=5, seed=3)
    td[td["Date"].dt.year == 2020].to_excel(raw / "2020.xlsx", index=False)
    td[td["Date"].dt.year == 2021].to_csv(raw / "2021.csv", index=False)
    sack.to_csv(raw / "sackmann" / "atp_matches_2020.csv", index=False)
    return raw


def _cfg(**kw):
    base = dict(start=pd.Timestamp("2021-01-01"), refit_days=30, blend_min=200, burn_in_days=120,
                staking=StakingConfig(ev_min=0.03))
    base.update(kw)
    return TennisConfig(**base)


def test_max_drawdown():
    assert max_drawdown(np.array([100, 50, 150, 75])) == pytest.approx(0.5)


def test_prepare_is_walk_forward(data_dir):
    ab, info = prepare(data_dir, _cfg())
    assert ab["y"].between(0, 1).all() and 0.4 < ab["y"].mean() < 0.6
    assert ab["p_markov"].notna().sum() > 100
    assert 0.0 <= info["w"] <= 1.0
    # predictions before a cut do not change when later data is removed
    cut = pd.Timestamp("2020-12-01")
    tmp = data_dir.parent / "cut" / "raw"
    (tmp / "sackmann").mkdir(parents=True, exist_ok=True)
    td = pd.concat([pd.read_excel(data_dir / "2020.xlsx"), pd.read_csv(data_dir / "2021.csv", parse_dates=["Date"])])
    td[td["Date"] < cut].to_csv(tmp / "all.csv", index=False)
    sack = pd.read_csv(data_dir / "sackmann" / "atp_matches_2020.csv")
    sack.to_csv(tmp / "sackmann" / "atp_matches_2020.csv", index=False)
    ab_cut, _ = prepare(tmp, _cfg(w=info["w"]))
    ab_full, _ = prepare(data_dir, _cfg(w=info["w"]))
    key = ["date", "winner", "loser"]
    merged = ab_full[ab_full["date"] < cut].merge(ab_cut, on=key, suffixes=("", "_cut"))
    assert len(merged) == (ab_full["date"] < cut).sum()
    np.testing.assert_allclose(np.where(merged["y"] == merged["y_cut"], merged["p_elo"], 1 - merged["p_elo"]),
                               merged["p_elo_cut"])
    both = merged["p_markov"].notna()
    np.testing.assert_allclose(np.where(merged["y"] == merged["y_cut"], merged["p_markov"], 1 - merged["p_markov"])[both],
                               merged["p_markov_cut"][both])


def test_filters(data_dir):
    ab, _ = prepare(data_dir, _cfg())
    rows = evaluation_rows(ab, _cfg(surface="clay", series="Grand Slam"))
    assert len(rows) > 0
    assert (rows["surface"] == "Clay").all() and (rows["level"] == "Grand Slam").all()
    assert (rows["date"] >= pd.Timestamp("2021-01-01")).all()


def test_run_backtest_outputs(data_dir, tmp_path):
    summary = run_backtest(data_dir, _cfg(), tmp_path)
    for name in ("bets.csv", "comparison.csv", "predictions.csv", "summary.json"):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    sources = {r["source"] for r in saved["comparison"]}
    assert {"elo", "markov", "blend_elo", "market_bet_time", "pinnacle_close"} <= sources
    assert any("piața" in v for v in summary["verdicts"])
    pin = next(r for r in saved["comparison"] if r["source"] == "pinnacle_close")
    preds = pd.read_csv(tmp_path / "predictions.csv")
    assert pin["n"] == int((preds["pin_a"].notna() & preds["q_a"].notna()).sum())
    assert pin["market_log_loss"] is not None  # market scored on the same matches as Pinnacle
    assert set(summary["by_surface"]) <= {"Hard", "Clay", "Grass"}
    assert "Grand Slam" in summary["by_level"]
    bets = pd.read_csv(tmp_path / "bets.csv", parse_dates=["date"])
    assert (bets["date"] >= pd.Timestamp("2021-01-01")).all()


def test_cli(data_dir, tmp_path):
    runner = CliRunner()
    model = tmp_path / "elo.pkl"
    r = runner.invoke(app, ["fit", "--data", str(data_dir), "--out", str(model)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["predict", "--model", str(model), "--player-a", "Alpha J.", "--player-b", "Bravo J.",
                            "--surface", "Clay", "--odds-a", "1.8", "--odds-b", "2.1"])
    assert r.exit_code == 0, r.output
    assert "elo" in r.output and "Piață" in r.output
    r = runner.invoke(app, ["backtest", "--data", str(data_dir), "--start", "2021-01-01", "--surface", "clay",
                            "--out-dir", str(tmp_path / "bt")])
    assert r.exit_code == 0, r.output
    assert "Comparație" in r.output
