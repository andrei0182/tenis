import numpy as np
import pandas as pd
import pytest

from tenis.blend import MarketBlender, devig, devig_power, devig_proportional, logit
from tenis.staking import StakingConfig, best_outcome, expected_value, kelly_fraction, simulate_bankroll, stake_size


def test_devig_proportional():
    p = devig_proportional(np.array([[2.0, 3.5, 4.0]]))
    np.testing.assert_allclose(p.sum(), 1.0)
    q = np.array([1 / 2.0, 1 / 3.5, 1 / 4.0])
    np.testing.assert_allclose(p[0], q / q.sum())


def test_devig_power_sums_to_one_and_shifts_to_favourite():
    odds = np.array([[1.3, 5.0, 9.0], [1.9, 1.9, np.nan]])
    p = devig_power(odds)
    assert p[0].sum() == pytest.approx(1.0)
    assert np.isnan(p[1]).all()
    # power method takes more margin from longshots than proportional
    assert p[0, 0] > devig_proportional(odds[:1])[0, 0]
    assert devig(odds[:1], "power")[0, 2] < devig(odds[:1], "proportional")[0, 2]


def test_logit_clips():
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def _binary_data(n, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) > true).astype(int)  # 0 = outcome 0 happened
    market = np.clip(true + rng.normal(0, 0.03, n), 0.05, 0.95)
    model = np.clip(true + rng.normal(0, 0.15, n), 0.05, 0.95)
    to2 = lambda p: np.column_stack([p, 1 - p])
    return to2(model), to2(market), y


def test_blender_prefers_sharper_market():
    pm, pk, y = _binary_data(4000)
    b = MarketBlender().fit(pm, pk, y)
    assert b.is_fitted
    c = b.coefficients
    assert c["w_market"] > c["w_model"]
    out = b.predict(pm, pk)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_blender_needs_min_rows_and_handles_nan():
    pm, pk, y = _binary_data(50)
    b = MarketBlender(min_rows=200).fit(pm, pk, y)
    assert not b.is_fitted
    assert np.isnan(b.predict(pm, pk)).all()
    pm, pk, y = _binary_data(500)
    pk[0] = np.nan
    b = MarketBlender(min_rows=200).fit(pm, pk, y)
    out = b.predict(pm, pk)
    assert np.isnan(out[0]).all() and np.isfinite(out[1:]).all()


def test_blender_three_way_sums_to_one():
    rng = np.random.default_rng(1)
    p = rng.dirichlet([3, 2, 2], 600)
    y = np.array([rng.choice(3, p=row) for row in p])
    b = MarketBlender().fit(p, p, y)
    np.testing.assert_allclose(b.predict(p, p).sum(axis=1), 1.0)


def test_ev_and_kelly():
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    assert kelly_fraction(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    assert kelly_fraction(0.3, 2.0) == 0.0


def test_stake_is_capped():
    cfg = StakingConfig(kelly=0.25, cap=0.02)
    assert stake_size(0.5, 2.2, 1000, cfg) == pytest.approx(0.02 * 1000)  # 0.25*0.0833 > 0.02
    assert stake_size(0.36, 3.0, 1000, cfg) == pytest.approx(0.25 * 0.04 * 1000)


def test_best_outcome_respects_ev_min():
    p = np.array([[0.5, 0.3, 0.2], [0.34, 0.33, 0.33]])
    odds = np.array([[2.2, 3.0, 4.0], [2.9, 3.0, 3.0]])
    idx, ev = best_outcome(p, odds, 0.03)
    assert idx.tolist() == [0, -1]
    assert ev[0] == pytest.approx(0.1)


def test_simulate_bankroll_same_day_uses_start_bankroll():
    bets = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
        "p": [0.6, 0.6, 0.6], "odds": [2.0, 2.0, 2.0], "won": [True, False, True],
    })
    out = simulate_bankroll(bets, StakingConfig(kelly=0.25, cap=0.02, bankroll=1000))
    assert out["stake"].iloc[0] == pytest.approx(20) and out["stake"].iloc[1] == pytest.approx(20)
    assert out["bankroll_after_day"].iloc[1] == pytest.approx(1000)
    assert out["stake"].iloc[2] == pytest.approx(20)
    assert out["bankroll_after_day"].iloc[2] == pytest.approx(1020)
