"""Load and clean tennis-data.co.uk results/odds and Jeff Sackmann's match stats."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_ORDER = {
    "0th Round": 0, "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Round Robin": 4, "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
}
SURFACES = ("Hard", "Clay", "Grass")

# Bet-time odds (model/blend input + the price we take); preferred first.
ODDS_SOURCES = {"Avg": ("AvgW", "AvgL"), "B365": ("B365W", "B365L")}
# Pinnacle: reference market and CLV only; never a model or blend input.
CLOSING = ("PSW", "PSL")

LEVELS = {
    "grand slam": "Grand Slam",
    "masters": "Masters", "masters 1000": "Masters", "wta1000": "Masters",
    "premier mandatory": "Masters", "premier 5": "Masters", "tier i": "Masters",
    "atp500": "ATP 500", "international gold": "ATP 500", "wta500": "ATP 500", "premier": "ATP 500",
    "tier ii": "ATP 500",
    "atp250": "ATP 250", "international": "ATP 250", "wta250": "ATP 250",
    "tier iii": "ATP 250", "tier iv": "ATP 250",
    "masters cup": "Tour Finals", "tour championships": "Tour Finals",
}


def normalize_player(name: object) -> str:
    """Unique player key: 'Djokovic N.' -> 'djokovic n', 'Auger-Aliassime F.' -> 'auger aliassime f'."""
    text = str(name).lower().replace(".", " ").replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_surface(value: object) -> str:
    """Hard / Clay / Grass; carpet is treated as a fast indoor hard court."""
    s = str(value).strip().capitalize()
    return "Hard" if s == "Carpet" else s


def series_level(value: object) -> str:
    """Map tennis-data Series/Tier names (ATP and WTA, old and new) to one level."""
    return LEVELS.get(str(value).strip().lower(), str(value).strip() or "Other")


def parse_dates(values: pd.Series) -> pd.Series:
    """Excel datetimes as-is; text as ISO (yyyy-mm-dd) or day-first (dd/mm/yyyy)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = values.astype(str).str.strip().str.slice(0, 10)
    iso = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    dmy_short = pd.to_datetime(text, format="%d/%m/%y", errors="coerce")
    return iso.fillna(dmy).fillna(dmy_short)


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _odds(raw: pd.DataFrame, source: str) -> tuple[pd.Series, pd.Series]:
    """Bet-time odds from `source`, falling back to the other source per row."""
    order = [source] + [s for s in ODDS_SOURCES if s != source]
    w = pd.Series(np.nan, index=raw.index)
    lo = pd.Series(np.nan, index=raw.index)
    for s in order:
        cw, cl = ODDS_SOURCES[s]
        if cw in raw.columns and cl in raw.columns:
            both = w.isna() | lo.isna()
            w = w.where(~both, pd.to_numeric(raw[cw], errors="coerce"))
            lo = lo.where(~both, pd.to_numeric(raw[cl], errors="coerce"))
    return w, lo


OVERROUND_RANGE = (1.0, 1.2)


def sane_odds_pair(w: pd.Series, lo: pd.Series) -> tuple[pd.Series, pd.Series]:
    """NaN for pairs that cannot be real prices: odds <= 1 or 1/w + 1/l outside OVERROUND_RANGE."""
    w, lo = w.where(w > 1.0), lo.where(lo > 1.0)
    overround = 1.0 / w + 1.0 / lo
    ok = overround.between(*OVERROUND_RANGE)
    return w.where(ok), lo.where(ok)


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(np.nan, index=raw.index)


def clean_tennis_data(raw: pd.DataFrame, odds_source: str = "Avg", tour: str = "ATP") -> pd.DataFrame:
    """Standardise one tennis-data.co.uk sheet."""
    series_col = "Series" if "Series" in raw.columns else "Tier"
    odds_w, odds_l = sane_odds_pair(*_odds(raw, odds_source))
    close_w, close_l = sane_odds_pair(_num(raw, CLOSING[0]), _num(raw, CLOSING[1]))
    comment = raw["Comment"].astype(str).str.strip() if "Comment" in raw.columns else "Completed"
    df = pd.DataFrame({
        "date": parse_dates(raw["Date"]),
        "tour": tour,
        "tournament": raw["Tournament"].astype(str).str.strip(),
        "tournament_no": _num(raw, tour),
        "series": raw[series_col].astype(str).str.strip() if series_col in raw.columns else "",
        "surface": raw["Surface"].map(normalize_surface),
        "round": raw["Round"].astype(str).str.strip(),
        "best_of": _num(raw, "Best of").fillna(3).astype(int),
        "winner_name": raw["Winner"].astype(str).str.strip(),
        "loser_name": raw["Loser"].astype(str).str.strip(),
        "wrank": _num(raw, "WRank"), "lrank": _num(raw, "LRank"),
        "comment": comment,
        "odds_w": odds_w, "odds_l": odds_l,
        "close_w": close_w, "close_l": close_l,
    })
    df["level"] = df["series"].map(series_level)
    df["round_no"] = df["round"].map(ROUND_ORDER).fillna(1).astype(int)
    df["winner"] = df["winner_name"].map(normalize_player)
    df["loser"] = df["loser_name"].map(normalize_player)
    df["completed"] = df["comment"].str.lower().eq("completed")
    return df.dropna(subset=["date"])


def _tour_of(path: Path, raw: pd.DataFrame) -> str:
    if "WTA" in raw.columns or "wta" in path.name.lower():
        return "WTA"
    return "ATP"


def load_tennis(data_dir: str | Path, odds_source: str = "Avg") -> pd.DataFrame:
    """Load every tennis-data .xlsx/.csv in `data_dir` (not subfolders), sorted chronologically."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".xlsx", ".xls", ".csv") and p.name != "player_map.csv")
    if not files:
        raise FileNotFoundError(f"no tennis-data files in {data_dir}")
    frames = []
    for path in files:
        raw = _read_any(path)
        frames.append(clean_tennis_data(raw, odds_source, _tour_of(path, raw)))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "tour", "tournament", "winner", "loser"], keep="last")
    df = df.sort_values(["date", "tour", "tournament_no", "tournament", "round_no"], kind="stable")
    return df.reset_index(drop=True)


def to_ab(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Random A/B orientation (fixed seed) so the winner is not always first. y = 1 if A won."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5
    out = df.copy()
    pairs = [("player_a", "player_b", "winner", "loser"), ("name_a", "name_b", "winner_name", "loser_name"),
             ("rank_a", "rank_b", "wrank", "lrank"), ("odds_a", "odds_b", "odds_w", "odds_l"),
             ("close_a", "close_b", "close_w", "close_l")]
    for a, b, w, lo in pairs:
        out[a] = np.where(flip, df[lo], df[w])
        out[b] = np.where(flip, df[w], df[lo])
    out["y"] = (~flip).astype(int)
    return out


# --- Jeff Sackmann serve/return statistics (Phase 2) -------------------------

SACKMANN_COLUMNS = ["tourney_date", "tourney_name", "surface", "best_of", "winner_name", "loser_name", "score",
                    "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]


def load_sackmann(folder: str | Path) -> pd.DataFrame | None:
    """Load `*_matches_*.csv` from tennis_atp / tennis_wta; drop matches without serve stats or not completed."""
    folder = Path(folder)
    files = sorted(folder.glob("*matches_*.csv")) if folder.exists() else []
    if not files:
        return None
    raw = pd.concat([pd.read_csv(p, low_memory=False) for p in files], ignore_index=True)
    df = raw[[c for c in SACKMANN_COLUMNS if c in raw.columns]].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["surface"] = df["surface"].map(normalize_surface)
    score = df["score"].astype(str)
    unfinished = score.str.contains("RET|W/O|DEF|ABD|unfinished", case=False, regex=True)
    df = df[~unfinished].dropna(subset=["tourney_date", "w_svpt", "l_svpt"])
    df = df[(df["w_svpt"] > 0) & (df["l_svpt"] > 0)]
    df["w_sp_won"] = df["w_1stWon"] + df["w_2ndWon"]
    df["l_sp_won"] = df["l_1stWon"] + df["l_2ndWon"]
    return df.sort_values("tourney_date", kind="stable").reset_index(drop=True)


def sackmann_candidates(full_name: str) -> list[str]:
    """tennis-data style keys a Sackmann full name might map to.

    'Novak Djokovic' -> 'djokovic n'; 'Juan Martin del Potro' -> 'del potro jm', 'del potro j', ...
    """
    tokens = normalize_player(full_name).split()
    out = []
    for k in range(len(tokens) - 1, 0, -1):
        first, last = tokens[:k], " ".join(tokens[k:])
        out.append(f"{last} {''.join(t[0] for t in first)}")
        out.append(f"{last} {first[0][0]}")
    return list(dict.fromkeys(out))


def build_player_map(td_keys: set[str], sackmann_names: set[str],
                     map_csv: str | Path | None = None) -> tuple[dict[str, str], list[str]]:
    """Map Sackmann full names to tennis-data keys; returns (mapping, unmatched names).

    Explicit rows in `map_csv` (columns sackmann_name, td_name) win over automatic matching.
    """
    explicit: dict[str, str] = {}
    if map_csv is not None and Path(map_csv).exists():
        m = pd.read_csv(map_csv)
        explicit = {str(s).strip(): normalize_player(t) for s, t in zip(m["sackmann_name"], m["td_name"])}
    mapping, unmatched = {}, []
    for name in sorted(sackmann_names):
        if name in explicit:
            mapping[name] = explicit[name]
            continue
        hit = next((c for c in sackmann_candidates(name) if c in td_keys), None)
        if hit is None:
            unmatched.append(name)
        else:
            mapping[name] = hit
    return mapping, unmatched
