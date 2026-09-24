import numpy as np
import pandas as pd
import pytest

from tenis.blend import devig
from tenis.daily import report_html, run_range, weekly_html
from tenis.pinnacle import api_key, closing_lines, events_to_frame, snapshot
from tenis.staking import StakingConfig
from tenis.value import find_value, join_sources, player_key, settle, summarize, superbet_frame

NOW = pd.Timestamp("2026-09-23T08:00:00Z")


def _ev(eid, home, away, starts, ml, parent=None, league="ATP Hangzhou - R1"):
    return {"event_id": eid, "home": home, "away": away, "starts": starts, "league_name": league, "parent_id": parent,
            "periods": {"num_0": {"status": "open", "money_line": {"home": ml[0], "away": ml[1], "draw": None}}}}


def _payload(p1=2.29):
    return {"events": [
        _ev(1, "Yunchaokete Bu", "Michael Zheng", "2026-09-24T08:00:00Z", (p1, 1.69)),
        _ev(2, "Yunchaokete Bu (Games)", "Michael Zheng (Games)", "2026-09-24T08:00:00Z", (1.9, 1.9), parent=1),
        _ev(3, "Bencic / Golubic", "Siniakova / Krejcikova", "2026-09-24T09:00:00Z", (2.0, 1.8)),
        _ev(4, "Jie Cui", "Adolfo Daniel Vallejo", "2026-09-24T22:30:00Z", (3.55, 1.3356)),
    ]}


def _superbet(date="2026-09-24", bu=2.45):
    return superbet_frame([
        {"tour": "atp", "tournament": "Hangzhou", "player1": "Michael Zheng", "player2": "Bu Yunchaokete",
         "odds_1": 1.64, "odds_2": bu},  # reversed order and reversed name
        {"tour": "atp", "tournament": "Hangzhou", "player1": "Jie Cui", "player2": "Adolfo Daniel Vallejo",
         "odds_1": 3.25, "odds_2": 1.32},
        {"tour": "atp", "tournament": "X", "player1": "Nobody Here", "player2": "Someone Else",
         "odds_1": 2.0, "odds_2": 1.8},
        {"tour": "wta", "tournament": "X", "player1": "Bencic B. / Golubic V.", "player2": "S / K",
         "odds_1": 2.0, "odds_2": 1.8},
    ], date)


def test_player_key_ignores_order_accents_case():
    assert player_key("Bu Yunchaokete") == player_key("Yunchaokete Bu")
    assert player_key("Jiří Lehečka") == player_key("jiri lehecka")


def test_events_to_frame_keeps_singles_main_events():
    df = events_to_frame(_payload(), NOW)
    assert df["Player1"].tolist() == ["Yunchaokete Bu", "Jie Cui"]
    assert df.iloc[1]["Date"] == "2026-09-25"  # 22:30 UTC is already the next day in Bucharest


def test_join_handles_swapped_players_next_day_and_price_check(tmp_path):
    sharp = events_to_frame(_payload(), NOW)
    sb = _superbet()
    assert len(sb) == 3  # doubles dropped
    pairs, unmatched = join_sources(sharp, sb)
    bu = pairs[pairs["Player1"] == "Yunchaokete Bu"].iloc[0]
    assert (bu["SB1"], bu["SB2"]) == (2.45, 1.64)  # Superbet odds realigned to Pinnacle's order
    assert "Jie Cui" in pairs["Player1"].tolist()  # Superbet day 24, Pinnacle local day 25: still paired
    assert unmatched["player1"].tolist() == ["Nobody Here"]
    wrong = sb.assign(SB1=[1.05, 3.25, 2.0], SB2=[12.0, 1.32, 1.8])  # prices far from Pinnacle's -> rejected
    pairs2, unmatched2 = join_sources(sharp, wrong)
    assert "Yunchaokete Bu" not in pairs2["Player1"].tolist() and len(unmatched2) == 2


def test_find_value_and_settle():
    sharp = events_to_frame(_payload(), NOW)
    pairs, _ = join_sources(sharp, _superbet())
    bets = find_value(pairs, StakingConfig(), ev_min=0.02)
    fair = devig(np.array([[2.29, 1.69]]), "power")[0]
    assert bets["player"].tolist() == ["Yunchaokete Bu"]
    assert bets.iloc[0]["ev"] == pytest.approx(2.45 * fair[0] - 1)
    closing = pd.DataFrame({"event_id": [1], "Player1": ["Yunchaokete Bu"], "PSC1": [2.1], "PSC2": [1.8]})
    s = settle(bets, closing)
    assert s.iloc[0]["clv"] == pytest.approx(2.45 / 2.1 - 1)
    assert s.iloc[0]["ev_close"] == pytest.approx(2.45 * devig(np.array([[2.1, 1.8]]), "power")[0, 0] - 1)
    assert summarize(s)["with_closing_odds"] == 1


def test_snapshot_history_and_closing(tmp_path):
    hist = tmp_path / "h.csv"
    snapshot(tmp_path / "s.csv", hist, now=NOW, payload=_payload(2.29))
    snapshot(tmp_path / "s.csv", hist, now=NOW + pd.Timedelta(hours=20), payload=_payload(2.1))
    close = closing_lines(hist).set_index("Player1")
    assert close.loc["Yunchaokete Bu", "PSC1"] == 2.1
    assert len(pd.read_csv(hist)) == 2  # compacted to one row per match


def test_run_range_reports_and_weekly(tmp_path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setattr("tenis.pinnacle.pd.Timestamp.now", lambda tz=None: NOW)
    res = run_range({"2026-09-24": _superbet()}, state, StakingConfig(), payload=_payload(2.29))
    r = res["2026-09-24"]
    assert r.compared == 2 and len(r.bets) == 1 and len(r.unmatched) == 1
    subject, html = report_html(res)
    assert subject == "Value bets tenis (1) -- 24.09.2026" and "Yunchaokete Bu" in html
    assert "<td>11:00</td>" in html and "ATP Hangzhou - R1" in html  # 08:00 UTC = 11:00 Bucharest

    monkeypatch.setattr("tenis.pinnacle.pd.Timestamp.now", lambda tz=None: NOW + pd.Timedelta(hours=20))
    res = run_range({"2026-09-24": _superbet(bu=1.5), "2026-09-25": _superbet("2026-09-25", bu=1.5)},
                    state, StakingConfig(), payload=_payload(2.1))
    assert res["2026-09-25"].summary["with_closing_odds"] == 1
    subject, html = report_html(res)
    assert "24.09-25.09.2026" in subject and "Bilanț de la început" in html

    subject, html = weekly_html(state, "2026-09-28")
    assert "(1 pariuri)" in subject and "Zile rulate: 2 din 7" in html and "2.10" in html


def test_weekly_empty_state(tmp_path):
    assert "Încă niciun pariu" in weekly_html(tmp_path, "2026-09-28")[1]


def test_api_key_required(monkeypatch):
    monkeypatch.delenv("PINNAPI_KEY", raising=False)
    with pytest.raises(RuntimeError):
        api_key()
