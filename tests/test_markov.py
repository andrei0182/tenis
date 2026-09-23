import numpy as np
import pandas as pd
import pytest

from tenis.markov import ServeReturnModel, markov_predictions, match_distribution, p_game, p_match, p_set, \
    p_tiebreak, set_score_distribution


def test_game_formula_known_values():
    assert p_game(0.5) == pytest.approx(0.5)
    assert p_game(0.6) == pytest.approx(0.7357, abs=1e-4)  # textbook value
    assert p_game(1.0) == pytest.approx(1.0)


def test_game_matches_simulation():
    rng = np.random.default_rng(0)
    p, wins, n = 0.63, 0, 20000
    for _ in range(n):
        a = b = 0
        while True:
            if rng.random() < p:
                a += 1
            else:
                b += 1
            if a >= 4 and a - b >= 2:
                wins += 1
                break
            if b >= 4 and b - a >= 2:
                break
    assert wins / n == pytest.approx(p_game(p), abs=0.01)


def test_symmetric_players_give_half():
    assert p_tiebreak(0.5, 0.5) == pytest.approx(0.5)
    assert p_set(0.5, 0.5) == pytest.approx(0.5)
    assert p_match(0.5, 0.5, 3) == pytest.approx(0.5)
    assert p_match(0.5, 0.5, 5) == pytest.approx(0.5)
    assert p_match(0.64, 0.64, 3) == pytest.approx(0.5)


def test_monotone_in_p():
    ps = np.linspace(0.5, 0.8, 7)
    probs = [p_match(p, 0.62) for p in ps]
    assert np.all(np.diff(probs) > 0)
    assert np.all(np.diff([p_game(p) for p in ps]) > 0)
    assert p_match(0.68, 0.62, 5) > p_match(0.68, 0.62, 3)  # longer match favours the better player


def test_distributions_sum_to_one():
    d = set_score_distribution(0.66, 0.6, True)
    assert sum(d.values()) == pytest.approx(1.0)
    assert set(d) >= {(6, 0), (7, 5), (7, 6), (6, 7)}
    m = match_distribution(0.66, 0.6, 3)
    assert sum(m.set_scores.values()) == pytest.approx(1.0)
    assert m.total_games.sum() == pytest.approx(1.0)
    assert m.p_match == pytest.approx(m.set_scores["2-0"] + m.set_scores["2-1"])
    assert m.p_set_handicap(-1.5) == pytest.approx(m.set_scores["2-0"])
    assert 0 < m.p_total_games_over(22.5) < 1
    assert m.total_games[:12].sum() == pytest.approx(0.0)  # a best-of-3 needs at least 12 games


def test_serve_return_estimates_and_no_leakage():
    sack = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2023-01-02"] * 3 + ["2023-02-06"]),
        "surface": "Hard", "winner_name": ["Big Server", "Big Server", "Big Server", "Weak Guy"],
        "loser_name": ["Weak Guy", "Weak Guy", "Weak Guy", "Big Server"],
        "w_sp_won": [60, 60, 60, 10], "w_svpt": [80, 80, 80, 80], "l_sp_won": [40, 40, 40, 10],
        "l_svpt": [80, 80, 80, 80],
    })
    names = {"Big Server": "server b", "Weak Guy": "guy w"}
    matches = pd.DataFrame({"date": pd.to_datetime(["2023-01-10", "2023-02-01", "2023-02-10"]),
                            "player_a": "server b", "player_b": "guy w", "surface": "Hard", "best_of": 3})
    model = ServeReturnModel(min_points=100, prior_points=10)
    p, model = markov_predictions(matches, sack, names, lag_days=15, model=model)
    assert np.isnan(p[0])  # tournament not finished yet: no stats available
    assert p[1] > 0.8  # big server dominated the earlier tournament
    # the Feb 6 tournament is not available until Feb 21: removing it changes nothing
    p_without, _ = markov_predictions(matches, sack.iloc[:3], names, lag_days=15,
                                      model=ServeReturnModel(min_points=100, prior_points=10))
    np.testing.assert_allclose(p, p_without, equal_nan=True)
    pa, pb = model.serve_probs("server b", "guy w", "Hard")
    assert pa > pb
