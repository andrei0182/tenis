import io
import json
import urllib.error

import pandas as pd
import pytest

from tenis import oddspapi, pinnacle


def _fixture(fid, p1, p2, start, tid=7, tname="ATP Challenger Bari"):
    return {"fixtureId": fid, "participant1Name": p1, "participant2Name": p2, "startTime": start,
            "tournamentId": tid, "tournamentName": tname, "statusId": 0, "hasOdds": True}


def _odds(fid, price1, price2, suspended=False):
    out = lambda p: {"players": {"0": {"active": True, "price": p}}}
    return {"fixtureId": fid, "bookmakerOdds": {"pinnacle": {
        "bookmakerIsActive": True, "suspended": suspended,
        "markets": {"121": {"marketActive": True, "outcomes": {"121": out(price1), "122": out(price2)}},
                    "1012": {"outcomes": {"1012": out(1.9), "1013": out(1.9)}}}}}}


FIX = [_fixture("id1", "Jannik Sinner", "Carlos Alcaraz", "2026-10-12T12:00:00.000Z"),
       _fixture("id2", "A. Smith / B. Jones", "C. Doe / D. Roe", "2026-10-12T13:00:00.000Z"),
       _fixture("id3", "Player One", "Player Two", "2026-10-12T14:00:00.000Z"),
       _fixture("id4", "No Odds", "At All", "2026-10-12T15:00:00.000Z", tid=8)]
ODDS = [_odds("id1", 1.85, 2.02), _odds("id2", 1.5, 2.6), _odds("id3", 1.4, 3.0, suspended=True)]


def test_to_frame_keeps_singles_with_active_match_winner():
    df = oddspapi.to_frame(FIX, ODDS)
    assert list(df["event_id"]) == ["id1"]
    r = df.iloc[0]
    assert (r.Player1, r.Player2, r.PS1, r.PS2, r.League) == ("Jannik Sinner", "Carlos Alcaraz", 1.85, 2.02,
                                                               "ATP Challenger Bari")


def test_fetch_one_fixtures_call_plus_chunked_odds(monkeypatch):
    calls = []

    def fake_get(path, params, timeout=60):
        calls.append((path, params))
        return FIX if path == "fixtures" else {"data": ODDS}

    monkeypatch.setattr(oddspapi, "_get", fake_get)
    monkeypatch.setattr(oddspapi, "CHUNK", 1)
    fixtures, odds = oddspapi.fetch("k", pd.Timestamp("2026-10-11T08:00:00Z"), sleep=lambda s: None)
    assert [c[0] for c in calls] == ["fixtures", "odds-by-tournaments", "odds-by-tournaments"]
    assert calls[0][1]["sportId"] == 12 and calls[0][1]["bookmakers"] == "pinnacle"
    assert calls[1][1]["tournamentIds"] == "7" and calls[2][1]["tournamentIds"] == "8"
    assert len(fixtures) == 4 and len(odds) == 6


def test_snapshot_uses_oddspapi_when_its_key_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("ODDSPAPI_KEY", "secret")
    monkeypatch.delenv("ODDS_PROVIDER", raising=False)
    monkeypatch.setattr(oddspapi, "fetch", lambda key, now, days=3: (FIX, ODDS))
    now = pd.Timestamp("2026-10-11T08:00:00Z")
    df = pinnacle.snapshot(tmp_path / "sharp.csv", tmp_path / "hist.csv", now=now)
    assert pinnacle.provider() == "oddspapi"
    assert list(df["event_id"]) == ["id1"] and df.iloc[0]["Date"] == "2026-10-12"
    closing = pinnacle.closing_lines(tmp_path / "hist.csv")
    assert closing.iloc[0]["PSC1"] == 1.85


def test_refused_key_raises_unavailable(monkeypatch):
    def refuse(req, timeout=60):
        body = io.BytesIO(json.dumps({"error": {"code": "QUOTA_EXCEEDED", "message": "limit"}}).encode())
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many", {}, body)

    monkeypatch.setattr(oddspapi.urllib.request, "urlopen", refuse)
    with pytest.raises(pinnacle.PinnapiUnavailable, match="QUOTA_EXCEEDED"):
        oddspapi._get("fixtures", {"apiKey": "k"})
