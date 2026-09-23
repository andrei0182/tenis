import numpy as np
import pandas as pd
import pytest

from tenis.elo import EloModel, EloRatings, combined_prob, k_factor, run_elo, select_w, set_prob_from_bo3, \
    to_best_of_5, win_prob


def test_probabilities_symmetric():
    ra, rb = np.array([1500, 1720, 1333]), np.array([1600, 1500, 1900])
    np.testing.assert_allclose(win_prob(ra, rb) + win_prob(rb, ra), 1.0)
    assert win_prob(1500, 1500) == pytest.approx(0.5)
    p = combined_prob(1600, 1500, 1450, 1700, 0.3)
    q = combined_prob(1500, 1600, 1700, 1450, 0.3)
    assert p + q == pytest.approx(1.0)


def test_k_decreases_with_experience():
    k = k_factor(np.arange(0, 500, 10))
    assert np.all(np.diff(k) < 0)
    assert k_factor(0) == pytest.approx(250 / 5 ** 0.4)


def test_winner_rating_increases_loser_decreases():
    r = EloRatings()
    r.update("a", "b", "Clay")
    g_a, s_a = r.ratings("a", "Clay")
    g_b, s_b = r.ratings("b", "Clay")
    assert g_a > 1500 and s_a > 1500 and g_b < 1500 and s_b < 1500
    assert r.ratings("a", "Grass") == (g_a, 1500.0)  # other surfaces untouched


def test_run_elo_uses_only_past_and_skips_incomplete():
    df = pd.DataFrame({"winner": ["a", "a", "b"], "loser": ["b", "c", "a"], "surface": ["Hard"] * 3,
                       "completed": [True, False, True]})
    out, ratings = run_elo(df)
    assert out.loc[0, "rg_w"] == 1500 and out.loc[0, "rg_l"] == 1500  # pre-match
    assert out.loc[1, "rg_w"] > 1500
    assert ratings.n_general["c"] == 0  # retirement did not update
    assert out.loc[2, "rg_w"] < 1500  # b lost the first match


def test_bo5_makes_favourite_stronger():
    for p in (0.55, 0.7, 0.9):
        assert to_best_of_5(p) > p
    assert to_best_of_5(0.5) == pytest.approx(0.5)
    assert to_best_of_5(0.3) < 0.3
    s = set_prob_from_bo3(0.7)
    assert s * s * (3 - 2 * s) == pytest.approx(0.7)


def test_select_w_prefers_informative_rating():
    rng = np.random.default_rng(0)
    n = 3000
    rs_a, rs_b = rng.normal(1500, 150, n), rng.normal(1500, 150, n)
    rg_a, rg_b = rng.normal(1500, 150, n), rng.normal(1500, 150, n)
    y = (rng.random(n) < win_prob(rs_a, rs_b)).astype(int)  # truth driven by surface rating only
    w, losses = select_w(rg_a, rg_b, rs_a, rs_b, y)
    assert w >= 0.8 and len(losses) == 11


def test_elo_model_predict():
    _, r = run_elo(pd.DataFrame({"winner": ["a"] * 5, "loser": ["b"] * 5, "surface": ["Clay"] * 5,
                                 "completed": [True] * 5}))
    m = EloModel(r.to_plain(), w=0.5)
    p3 = m.predict("a", "b", "Clay")
    assert p3 > 0.5 and m.predict("a", "b", "Clay", best_of=5) > p3
    assert m.predict("a", "b", "Clay") + m.predict("b", "a", "Clay") == pytest.approx(1.0)
