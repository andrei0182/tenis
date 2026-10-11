"""Pinnacle prematch tennis odds via oddspapi.io (match winner), as a replacement for pinnapi.com.

Pinnacle is only read as the reference ("fair") price; bets are placed at Superbet.
The API key is read from the ODDSPAPI_KEY environment variable and never written anywhere.

Cost per snapshot (free plan: 250 requests/month): 1 request for the fixtures list plus 1 per
`CHUNK` tournaments for the odds.

    python -m tenis.oddspapi --probe   # print the Pinnacle markets of one fixture (to check MARKET/OUTCOMES)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .pinnacle import PinnapiUnavailable

BASE_URL = "https://api.oddspapi.io/v4"
TENNIS = 12
BOOKMAKER = "pinnacle"
# Tennis match winner: market 121, outcomes 121 (participant 1) / 122 (participant 2), per oddspapi's tennis guide.
MARKET = "121"
OUTCOMES = ("121", "122")
CHUNK = 100  # tournament ids per odds request
COOLDOWN = 2.1  # seconds between calls (fixtures endpoint cooldown is 2000 ms)
COLUMNS = ["event_id", "starts", "League", "Player1", "Player2", "PS1", "PS2"]


class OddspapiUnavailable(PinnapiUnavailable):
    """oddspapi refused the key (missing/invalid key, quota used up)."""


def api_key() -> str:
    key = os.environ.get("ODDSPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("ODDSPAPI_KEY is not set")
    return key


def _get(path: str, params: dict, timeout: int = 60):
    url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "tenis-value/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 402, 403, 429):
            raise
        try:
            err = json.loads(exc.read().decode() or "{}").get("error", {})
        except ValueError:
            err = {}
        raise OddspapiUnavailable(f"oddspapi HTTP {exc.code} {err.get('code', '')}: {err.get('message', '')}".strip()) from exc


def _items(payload) -> list[dict]:
    """Responses are either a list of fixtures or an object wrapping one."""
    if isinstance(payload, list):
        return payload
    for k in ("data", "fixtures", "odds"):
        if isinstance(payload.get(k), list):
            return payload[k]
    return [payload] if "fixtureId" in payload else []


def _price(market: dict, outcome: str) -> float | None:
    players = ((market.get("outcomes") or {}).get(outcome) or {}).get("players") or {}
    p = players.get("0") or next(iter(players.values()), None)
    if not p or p.get("active") is False:
        return None
    return p.get("price")


def fetch(key: str, now: pd.Timestamp, days: int = 3, sleep=time.sleep) -> tuple[list[dict], list[dict]]:
    """(fixtures, odds) for prematch tennis fixtures with Pinnacle odds in the next `days`."""
    fixtures = _items(_get("fixtures", {
        "sportId": TENNIS, "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": (now + pd.Timedelta(days=min(days, 9))).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "statusId": 0, "hasOdds": "true", "bookmakers": BOOKMAKER, "apiKey": key}))
    tids = sorted({f["tournamentId"] for f in fixtures if f.get("tournamentId") is not None})
    odds = []
    for i in range(0, len(tids), CHUNK):
        sleep(COOLDOWN)
        odds += _items(_get("odds-by-tournaments", {
            "tournamentIds": ",".join(map(str, tids[i:i + CHUNK])), "bookmakers": BOOKMAKER, "apiKey": key}))
    return fixtures, odds


def to_frame(fixtures: list[dict], odds: list[dict]) -> pd.DataFrame:
    """Same columns as pinnacle.events_to_frame: singles match-winner prices, one row per fixture."""
    by_id = {o.get("fixtureId"): o for o in odds}
    rows = []
    for f in fixtures:
        name1, name2 = f.get("participant1Name"), f.get("participant2Name")
        if not name1 or not name2 or "/" in name1 or "/" in name2:  # doubles: "A / B"
            continue
        book = ((by_id.get(f["fixtureId"]) or {}).get("bookmakerOdds") or {}).get(BOOKMAKER) or {}
        if not book or book.get("suspended") or book.get("bookmakerIsActive") is False:
            continue
        market = (book.get("markets") or {}).get(MARKET) or {}
        if market.get("marketActive") is False:
            continue
        p1, p2 = _price(market, OUTCOMES[0]), _price(market, OUTCOMES[1])
        if not p1 or not p2:
            continue
        rows.append({"event_id": f["fixtureId"], "starts": f["startTime"], "League": f.get("tournamentName"),
                     "Player1": name1, "Player2": name2, "PS1": p1, "PS2": p2})
    return pd.DataFrame(rows, columns=COLUMNS)


def probe() -> None:
    """Print the Pinnacle markets of the first fixture with odds, to confirm MARKET / OUTCOMES."""
    now = pd.Timestamp.now(tz="UTC")
    fixtures, odds = fetch(api_key(), now, days=1)
    print(f"{len(fixtures)} fixtures, {len(odds)} with odds")
    for o in odds[:1]:
        f = next((x for x in fixtures if x["fixtureId"] == o.get("fixtureId")), {})
        print(f.get("tournamentName"), "|", f.get("participant1Name"), "vs", f.get("participant2Name"))
        for mid, m in ((o.get("bookmakerOdds") or {}).get(BOOKMAKER, {}).get("markets") or {}).items():
            prices = {oid: _price(m, oid) for oid in (m.get("outcomes") or {})}
            print(f"  market {mid}: {prices}")
    print(to_frame(fixtures, odds).head(10).to_string())


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
