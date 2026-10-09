"""Value bets on Superbet tennis (match winner) against Pinnacle's no-vig price, tracked by closing line value.

No model: the fair probability is the de-vigged Pinnacle price. A Superbet price is a value bet when
price * fair - 1 >= ev_min. Closing Pinnacle odds only evaluate bets (CLV, closing EV), never pick them.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .blend import devig
from .staking import StakingConfig, stake_size

KEY = ["date", "player", "opponent"]
LOG_COLUMNS = ["date", "tournament", "player", "opponent", "event_id", "odds", "fair_p", "fair_odds", "ev", "stake",
               "kickoff", "logged_at"]
LOCAL_TZ = "Europe/Bucharest"


def _kickoff_local(starts: object) -> str:
    """Local (Bucharest) HH:MM from Pinnacle's UTC start, empty when unknown."""
    ts = pd.to_datetime(starts, utc=True, errors="coerce")
    return "" if pd.isna(ts) else ts.tz_convert(LOCAL_TZ).strftime("%H:%M")


def player_key(name: object) -> str:
    """Loose player key: no accents, case or punctuation; word order ignored ('Bu Yunchaokete' == 'Yunchaokete Bu')."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9 ]", " ", text).split()
    return " ".join(sorted(words))


def _similarity(a: str, b: str) -> float:
    """1 when one name's words are all in the other (middle names, 'Wong' vs 'Chak Lam Coleman Wong'), else a ratio."""
    wa, wb = set(a.split()), set(b.split())
    if wa and wb and (wa <= wb or wb <= wa):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def load_superbet_tennis(path: str | Path, date: str) -> pd.DataFrame:
    """Superbet tennis scraper export (output/tenis.xlsx), singles only."""
    raw = pd.read_excel(path)
    df = pd.DataFrame({
        "date": pd.Timestamp(date), "tour": raw.get("Tur"), "tournament": raw.get("Turneu"),
        "player1": raw["Jucător 1"].astype(str).str.strip(), "player2": raw["Jucător 2"].astype(str).str.strip(),
        "SB1": pd.to_numeric(raw["Cotă 1"], errors="coerce"), "SB2": pd.to_numeric(raw["Cotă 2"], errors="coerce"),
    })
    doubles = raw["Dublu"].astype(str).str.lower().isin(["true", "1", "da", "yes"]) if "Dublu" in raw else False
    df = df[~doubles & ~df["player1"].str.contains("/") & (df["player2"] != "")]
    return df.reset_index(drop=True)


def superbet_frame(matches: list[dict], date: str) -> pd.DataFrame:
    """Rows {tour, tournament, player1, player2, odds_1, odds_2} from the Superbet API -> the loader's layout."""
    raw = pd.DataFrame(matches, columns=["tour", "tournament", "player1", "player2", "odds_1", "odds_2"])
    df = pd.DataFrame({"date": pd.Timestamp(date), "tour": raw["tour"], "tournament": raw["tournament"],
                       "player1": raw["player1"].astype(str).str.strip(), "player2": raw["player2"].astype(str).str.strip(),
                       "SB1": pd.to_numeric(raw["odds_1"], errors="coerce"),
                       "SB2": pd.to_numeric(raw["odds_2"], errors="coerce")})
    return df[~df["player1"].str.contains("/") & (df["player2"] != "")].reset_index(drop=True)


def _fair(p1: float, p2: float) -> np.ndarray:
    return devig(np.array([[p1, p2]], dtype=float), "power")[0]


def join_sources(sharp: pd.DataFrame, soft: pd.DataFrame, fuzzy: float = 0.8, max_gap: float = 0.15
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each Superbet match with a Pinnacle match on the same day (either player order).

    Returns (pairs with Pinnacle and Superbet odds aligned to Pinnacle's player order, unmatched Superbet rows).
    A pair is rejected when the two books' no-vig probabilities differ by more than `max_gap`.
    """
    sharp = sharp.assign(_1=sharp["Player1"].map(player_key), _2=sharp["Player2"].map(player_key),
                         _d=pd.to_datetime(sharp["Date"]))
    rows, used, unmatched = [], set(), []
    for _, sb in soft.iterrows():
        a, b = player_key(sb["player1"]), player_key(sb["player2"])
        best, best_score, swapped = None, fuzzy, False
        # Superbet's day is a UTC day, Pinnacle's a Bucharest day: late matches can sit one day apart
        near = sharp[(sharp["_d"] - sb["date"]).abs() <= pd.Timedelta(days=1)]
        for j, pin in near.iterrows():
            if j in used:
                continue
            for flip, (x, y) in ((False, (a, b)), (True, (b, a))):
                score = min(_similarity(x, pin["_1"]), _similarity(y, pin["_2"]))
                if score > best_score or (score == best_score and best is None):
                    best, best_score, swapped = j, score, flip
        if best is None:
            unmatched.append(sb)
            continue
        pin = sharp.loc[best]
        o1, o2 = (sb["SB2"], sb["SB1"]) if swapped else (sb["SB1"], sb["SB2"])
        if np.isfinite([o1, o2]).all() and np.max(np.abs(_fair(pin["PS1"], pin["PS2"]) - _fair(o1, o2))) > max_gap:
            unmatched.append(sb)
            continue
        used.add(best)
        rows.append({"date": pin["_d"], "tournament": pin["League"], "event_id": pin["event_id"],
                     "Player1": pin["Player1"], "Player2": pin["Player2"], "PS1": pin["PS1"], "PS2": pin["PS2"],
                     "SB1": o1, "SB2": o2, "superbet": f"{sb['player1']} - {sb['player2']}",
                     "kickoff": _kickoff_local(pin.get("starts"))})
    cols = ["date", "tournament", "event_id", "Player1", "Player2", "PS1", "PS2", "SB1", "SB2", "superbet", "kickoff"]
    unmatched_df = pd.DataFrame(unmatched)[["player1", "player2"]] if unmatched else pd.DataFrame(
        columns=["player1", "player2"])
    return pd.DataFrame(rows, columns=cols), unmatched_df.reset_index(drop=True)


def find_value(pairs: pd.DataFrame, staking: StakingConfig, ev_min: float = 0.02, max_odds: float = 8.0) -> pd.DataFrame:
    """Superbet prices at least `ev_min` above Pinnacle's no-vig price (power method), one row per bet."""
    out = []
    for r in pairs.itertuples():
        if not np.isfinite([r.PS1, r.PS2]).all():
            continue
        fair = _fair(r.PS1, r.PS2)
        for k, (player, opponent, odds) in enumerate(((r.Player1, r.Player2, r.SB1), (r.Player2, r.Player1, r.SB2))):
            if not np.isfinite(odds) or odds > max_odds:
                continue
            ev = odds * fair[k] - 1
            if ev >= ev_min:
                out.append({"date": r.date, "tournament": r.tournament, "player": player, "opponent": opponent,
                            "event_id": r.event_id, "odds": odds, "fair_p": fair[k], "fair_odds": 1 / fair[k], "ev": ev,
                            "stake": float(stake_size(fair[k], odds, staking.bankroll, staking)),
                            "kickoff": getattr(r, "kickoff", ""), "logged_at": None})
    return pd.DataFrame(out, columns=LOG_COLUMNS)


def read_log(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def append_log(bets: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Add bets to the log; a bet already logged keeps its first (earliest) price."""
    path = Path(path)
    if path.exists():
        bets = pd.concat([read_log(path), bets], ignore_index=True)
    bets = bets.drop_duplicates(subset=["event_id", "player"], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    bets.to_csv(path, index=False)
    return bets


def settle(log: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    """CLV and closing EV for each logged bet from the Pinnacle closing line of the same event."""
    cols = ["event_id", "Player1", "PSC1", "PSC2"] + (["taken_at"] if "taken_at" in closing else [])
    merged = log.merge(closing[cols].rename(columns={"taken_at": "close_taken_at"}), on="event_id", how="left")
    is_p1 = (merged["player"] == merged["Player1"]).to_numpy()
    close_odds = np.where(is_p1, merged["PSC1"], merged["PSC2"]).astype(float)
    fair = devig(merged[["PSC1", "PSC2"]].to_numpy(dtype=float), "power") if len(merged) else np.empty((0, 2))
    close_fair = np.where(is_p1, fair[:, 0], fair[:, 1]) if len(merged) else np.empty(0)
    out = merged.drop(columns=["Player1", "PSC1", "PSC2"])
    status = closing_status(out, close_odds)
    valid = status == "ok"
    out["closing_status"] = status
    out["close_odds"] = np.where(valid, close_odds, np.nan)
    out["clv"] = out["odds"] / out["close_odds"] - 1
    out["ev_close"] = np.where(valid, out["odds"] * close_fair - 1, np.nan)
    return out


MIN_CLOSING_GAP = pd.Timedelta(minutes=30)


def closing_status(settled: pd.DataFrame, close_odds: np.ndarray) -> np.ndarray:
    """Whether each bet's closing line really came after the bet.

    "ok": a closing price exists and, when timing is known, was taken at least 30 minutes after the bet;
    "fara_captura_ulterioara": the last pre-start snapshot is the one the bet was found in (no real close);
    "nemasurat": the bet predates `logged_at` tracking (timing unknown);
    "fara_inchidere": no closing price at all. Closing lines without snapshot times count as "ok".
    """
    has_close = np.isfinite(close_odds)
    status = np.where(has_close, "ok", "fara_inchidere").astype(object)
    if "close_taken_at" not in settled:
        return status
    taken = pd.to_datetime(settled["close_taken_at"], utc=True, errors="coerce", format="mixed")
    logged = (pd.to_datetime(settled["logged_at"], utc=True, errors="coerce", format="mixed")
              if "logged_at" in settled else pd.Series(pd.NaT, index=settled.index))
    unknown = has_close & logged.isna().to_numpy()
    too_early = has_close & ~unknown & (taken < logged + MIN_CLOSING_GAP).to_numpy()
    status[unknown] = "nemasurat"
    status[too_early] = "fara_captura_ulterioara"
    return status


def summarize(settled: pd.DataFrame) -> dict:
    """Closing EV (the edge estimate), CLV and counts."""
    ev = settled["ev_close"].dropna()
    clv = settled["clv"].dropna()
    status = settled["closing_status"] if "closing_status" in settled else pd.Series(dtype=object)
    return {"excluded_no_later_snapshot": int((status == "fara_captura_ulterioara").sum()),
            "excluded_unmeasured": int((status == "nemasurat").sum()),
            "bets": int(len(settled)), "with_closing_odds": int(len(ev)),
            "ev_close_mean": float(ev.mean()) if len(ev) else None,
            "ev_close_se": float(ev.std(ddof=1) / np.sqrt(len(ev))) if len(ev) > 1 else None,
            "clv_mean": float(clv.mean()) if len(clv) else None,
            "clv_positive_share": float((clv > 0).mean()) if len(clv) else None}
