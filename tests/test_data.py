import numpy as np
import pandas as pd

from tenis.data import build_player_map, load_sackmann, load_tennis, normalize_player, sackmann_candidates, \
    series_level, to_ab


def test_normalize_player():
    assert normalize_player("Djokovic N.") == "djokovic n"
    assert normalize_player(" Djokovic  N ") == "djokovic n"
    assert normalize_player("Auger-Aliassime F.") == "auger aliassime f"


def test_parse_dates():
    from tenis.data import parse_dates
    out = parse_dates(pd.Series(["2020-01-06", "06/01/2020", "06/01/20", "2020-01-06 00:00:00"]))
    assert (out == pd.Timestamp("2020-01-06")).all()


def test_series_level():
    assert series_level("Grand Slam") == "Grand Slam"
    assert series_level("Masters 1000") == "Masters"
    assert series_level("ATP250") == "ATP 250"
    assert series_level("WTA500") == "ATP 500"


def test_load_tennis(tmp_path):
    rows = [
        {"ATP": 2, "Tournament": "Rome", "Date": "2023-05-10", "Series": "Masters 1000", "Surface": "Clay",
         "Round": "2nd Round", "Best of": 3, "Winner": "Sinner J.", "Loser": "Alcaraz C.", "WRank": 4, "LRank": 1,
         "Comment": "Completed", "B365W": 2.1, "B365L": 1.7, "PSW": 2.2, "PSL": 1.75, "AvgW": 2.05, "AvgL": 1.72},
        {"ATP": 2, "Tournament": "Rome", "Date": "2023-05-10", "Series": "Masters 1000", "Surface": "Clay",
         "Round": "1st Round", "Best of": 3, "Winner": "Djokovic N.", "Loser": "Rune H.", "WRank": 2, "LRank": 8,
         "Comment": "Retired", "B365W": 1.3, "B365L": 3.5, "AvgW": np.nan, "AvgL": np.nan},
    ]
    pd.DataFrame(rows).to_excel(tmp_path / "2023.xlsx", index=False)
    df = load_tennis(tmp_path)
    assert list(df["round_no"]) == [1, 2]  # same day: earlier round first
    assert df.loc[0, "completed"] == False  # noqa: E712
    assert df.loc[0, "odds_w"] == 1.3  # fallback to B365 when Avg missing
    assert df.loc[1, "odds_w"] == 2.05 and df.loc[1, "close_w"] == 2.2
    assert df.loc[1, "level"] == "Masters"


def test_to_ab_is_random_but_consistent():
    df = pd.DataFrame({"winner": [f"w{i}" for i in range(200)], "loser": [f"l{i}" for i in range(200)],
                       "winner_name": "W", "loser_name": "L", "wrank": 1, "lrank": 2,
                       "odds_w": 1.5, "odds_l": 2.5, "close_w": 1.5, "close_l": 2.5})
    ab = to_ab(df, seed=1)
    assert 0.35 < ab["y"].mean() < 0.65
    won_a = ab["y"] == 1
    assert (ab.loc[won_a, "player_a"].str.startswith("w")).all()
    assert (ab.loc[~won_a, "player_b"].str.startswith("w")).all()
    assert (ab.loc[~won_a, "odds_b"] == 1.5).all()
    pd.testing.assert_frame_equal(ab, to_ab(df, seed=1))


def test_sackmann_mapping(tmp_path):
    assert "djokovic n" in sackmann_candidates("Novak Djokovic")
    assert "del potro jm" in sackmann_candidates("Juan Martin del Potro")
    pd.DataFrame({"sackmann_name": ["Carlos Alcaraz"], "td_name": ["Alcaraz Garfia C."]}).to_csv(
        tmp_path / "map.csv", index=False)
    mapping, unmatched = build_player_map({"djokovic n", "del potro jm"},
                                          {"Novak Djokovic", "Juan Martin del Potro", "Carlos Alcaraz", "Nobody Here"},
                                          tmp_path / "map.csv")
    assert mapping["Novak Djokovic"] == "djokovic n"
    assert mapping["Juan Martin del Potro"] == "del potro jm"
    assert mapping["Carlos Alcaraz"] == "alcaraz garfia c"
    assert unmatched == ["Nobody Here"]


def test_load_sackmann(tmp_path):
    base = {"tourney_name": "X", "surface": "Hard", "best_of": 3, "w_1stWon": 30, "w_2ndWon": 10,
            "l_1stWon": 25, "l_2ndWon": 8, "winner_name": "A B", "loser_name": "C D"}
    pd.DataFrame([{**base, "tourney_date": 20230102, "score": "6-4 6-4", "w_svpt": 60, "l_svpt": 62},
                  {**base, "tourney_date": 20230109, "score": "6-4 2-1 RET", "w_svpt": 30, "l_svpt": 30},
                  {**base, "tourney_date": 20230116, "score": "6-4 6-4", "w_svpt": np.nan, "l_svpt": 60}]
                 ).to_csv(tmp_path / "atp_matches_2023.csv", index=False)
    df = load_sackmann(tmp_path)
    assert len(df) == 1 and df.loc[0, "w_sp_won"] == 40
    assert load_sackmann(tmp_path / "missing") is None
