"""Pinnacle prematch tennis odds via pinnapi.com (match winner), with a snapshot history for closing lines.

The API key is read from the PINNAPI_KEY environment variable and never written anywhere.
The last snapshot taken before a match starts is its (approximate) closing line.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://pinnapi.com/kit/v1"
TENNIS = 2
LOCAL_TZ = "Europe/Bucharest"
COLUMNS = ["event_id", "starts", "League", "Player1", "Player2", "PS1", "PS2"]


def api_key() -> str:
    """PINNAPI_KEY from the environment (a GitHub/environment secret, never a file in the repo)."""
    key = os.environ.get("PINNAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("PINNAPI_KEY is not set")
    return key


def fetch_prematch(key: str, sport_id: int = TENNIS, timeout: int = 60) -> dict:
    """One REST call: every prematch tennis event (1 of the free tier's 100 requests/day)."""
    req = urllib.request.Request(f"{BASE_URL}/markets?sport_id={sport_id}&event_type=prematch",
                                 headers={"x-portal-apikey": key, "User-Agent": "tenis-value/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def is_doubles(name: str) -> bool:
    """Pinnacle writes doubles pairs as 'A / B'."""
    return "/" in str(name)


def events_to_frame(payload: dict, taken_at: pd.Timestamp) -> pd.DataFrame:
    """Match-winner odds for singles matches (games markets live on child events and are skipped)."""
    rows = []
    for ev in payload.get("events", []):
        game = (ev.get("periods") or {}).get("num_0") or {}
        ml = game.get("money_line") or {}
        if ev.get("parent_id") or game.get("status", "open") != "open" or not ml.get("home") or not ml.get("away"):
            continue
        if is_doubles(ev["home"]) or is_doubles(ev["away"]):
            continue
        rows.append({"event_id": ev["event_id"], "starts": ev["starts"], "League": ev.get("league_name"),
                     "Player1": ev["home"], "Player2": ev["away"], "PS1": ml["home"], "PS2": ml["away"]})
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["starts"] = pd.to_datetime(df["starts"], utc=True)
    df = df[df["starts"] > taken_at].copy()
    df.insert(0, "taken_at", taken_at)
    df.insert(1, "Date", df["starts"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d"))
    return df.reset_index(drop=True)


def compact_history(history: str | Path, keep_days: int = 45) -> pd.DataFrame:
    """Keep, per event, only the latest snapshot taken before the start; drop events older than `keep_days`."""
    h = pd.read_csv(history)
    columns = list(h.columns)  # appends are positional, so the column order must not change
    taken = pd.to_datetime(h["taken_at"], utc=True, format="mixed")
    starts = pd.to_datetime(h["starts"], utc=True, format="mixed")
    h = h[(taken < starts) & (starts >= taken.max() - pd.Timedelta(days=keep_days))]
    h = h.assign(_t=taken).sort_values("_t").groupby("event_id", as_index=False).last()
    h = h.sort_values(["_t", "event_id"])[columns]
    h.to_csv(history, index=False)
    return h


def snapshot(out: str | Path, history: str | Path | None = None, days: int = 3, key: str | None = None,
             now: pd.Timestamp | None = None, payload: dict | None = None) -> pd.DataFrame:
    """Write current prices for matches in the next `days` and add them to the (compacted) history."""
    now = now or pd.Timestamp.now(tz="UTC")
    payload = payload if payload is not None else fetch_prematch(key or api_key())
    df = events_to_frame(payload, now)
    df = df[df["starts"] <= now + pd.Timedelta(days=days)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["taken_at"]).to_csv(out, index=False)
    if history is not None:
        history = Path(history)
        history.parent.mkdir(parents=True, exist_ok=True)
        if history.exists():
            df = df[pd.read_csv(history, nrows=0).columns.tolist()]
        df.to_csv(history, mode="a", header=not history.exists(), index=False)
        compact_history(history)
    return df


def closing_lines(history: str | Path) -> pd.DataFrame:
    """Per event, the last pre-start snapshot as closing odds PSC1/PSC2."""
    h = pd.read_csv(history)
    h["taken_at"] = pd.to_datetime(h["taken_at"], utc=True, format="mixed")
    h["starts"] = pd.to_datetime(h["starts"], utc=True, format="mixed")
    h = h[h["taken_at"] < h["starts"]].sort_values("taken_at")
    last = h.groupby("event_id", as_index=False).last().rename(columns={"PS1": "PSC1", "PS2": "PSC2"})
    return last[["event_id", "Date", "League", "Player1", "Player2", "starts", "taken_at", "PSC1", "PSC2"]]
