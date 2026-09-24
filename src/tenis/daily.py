"""Daily / date-range / weekly value-betting reports: Pinnacle snapshot + Superbet tennis odds."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .pinnacle import closing_lines, snapshot
from .staking import StakingConfig
from .value import append_log, find_value, join_sources, read_log, settle, summarize

WEEKDAY = ["Lu", "Ma", "Mi", "Jo", "Vi", "Sâ", "Du"]
FOOTER = ("<p style='color:#666'>Estimare, nu garanție. EV la închidere pozitiv, constant, pe sute de pariuri e "
          "singurul semn credibil de avantaj. Pariază doar sume pe care îți permiți să le pierzi.</p>")


@dataclass
class DayResult:
    """One day's comparison."""

    bets: pd.DataFrame
    compared: int
    unmatched: pd.DataFrame
    summary: dict = field(default_factory=dict)


def run_range(superbet: dict[str, pd.DataFrame], state_dir: str | Path, staking: StakingConfig,
              ev_min: float = 0.02, max_odds: float = 8.0, payload: dict | None = None) -> dict[str, DayResult]:
    """One Pinnacle snapshot covering every date, then Superbet vs Pinnacle for each date."""
    state = Path(state_dir)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    days = max(3, (pd.Timestamp(max(superbet)) - today).days + 2)
    sharp = snapshot(state / "sharp.csv", state / "pinnacle_snapshots.csv", days=days, payload=payload)
    out = {}
    for date, sb in sorted(superbet.items()):
        pairs, unmatched = join_sources(sharp, sb)
        bets = find_value(pairs, staking, ev_min, max_odds)
        if not bets.empty:
            append_log(bets, state / "value_log.csv")
        record_day(state, date, len(pairs), len(unmatched), len(bets))
        out[date] = DayResult(bets, len(pairs), unmatched)
    settled = settle_log(state)
    for res in out.values():
        res.summary = summarize(settled) if settled is not None else {}
    return out


def record_day(state: Path, date: str, compared: int, unmatched: int, bets: int) -> None:
    """One row per day in daily_stats.csv (a re-run replaces the day)."""
    path = state / "daily_stats.csv"
    row = pd.DataFrame([{"date": date, "compared": compared, "unmatched": unmatched, "bets": bets}])
    if path.exists():
        old = pd.read_csv(path, dtype={"date": str})
        row = pd.concat([old[old["date"] != date], row], ignore_index=True)
    row.sort_values("date").to_csv(path, index=False)


def settle_log(state: Path) -> pd.DataFrame | None:
    """Every logged bet against its closing line; None when nothing is logged yet."""
    log_path, history = state / "value_log.csv", state / "pinnacle_snapshots.csv"
    if not log_path.exists() or not history.exists():
        return None
    settled = settle(read_log(log_path), closing_lines(history))
    settled.to_csv(state / "value_settled.csv", index=False)
    return settled


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _record_html(s: dict, title: str) -> str:
    """Closing-line record block; empty until at least one bet has a closing line."""
    if not s or not s.get("with_closing_odds"):
        return ""
    verdict = "clar pozitiv" if s["ev_close_se"] and s["ev_close_mean"] / s["ev_close_se"] > 2 else "încă neconcludent"
    return (f"<h3>{title}: {s['bets']} pariuri ({s['with_closing_odds']} cu linie de închidere)</h3>"
            f"<p>EV la închidere: <b>{_pct(s['ev_close_mean'])}</b> &plusmn; {_pct(s['ev_close_se'])} ({verdict}). "
            f"Pariuri cu CLV pozitiv: {_pct(s['clv_positive_share'])}.</p>")


def _bets_table(bets: pd.DataFrame, with_date: bool, closing: bool = False) -> str:
    e = html.escape
    if "kickoff" not in bets:  # older log rows predate this column
        bets = bets.assign(kickoff="")
    bets = bets.assign(kickoff=bets["kickoff"].fillna(""))
    head = (("<th>Data</th>" if with_date else "") + "<th>Ora</th><th>Turneu</th><th>Pariu pe</th><th>Cota Superbet</th>"
            "<th>Cota corectă (Pinnacle)</th><th>EV</th>"
            + ("<th>Cota Pinnacle la închidere</th><th>EV la închidere</th>" if closing else "<th>Miză sugerată</th>"))
    body = []
    for r in bets.sort_values(["date", "kickoff", "ev"], ascending=[True, True, False]).itertuples():
        cells = [f"{r.date:%d.%m}"] if with_date else []
        cells += [e(str(r.kickoff)), e(str(r.tournament)), f"<b>{e(r.player)}</b> vs {e(r.opponent)}", f"{r.odds:.2f}",
                  f"{r.fair_odds:.2f}", f"{r.ev * 100:+.1f}%"]
        if closing:
            cells += ["-" if pd.isna(r.close_odds) else f"{r.close_odds:.2f}",
                      "-" if pd.isna(r.ev_close) else f"{r.ev_close * 100:+.1f}%"]
        else:
            cells.append(f"{r.stake:.2f}")
        body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f"<table border='1' cellpadding='4' cellspacing='0'><tr>{head}</tr>{''.join(body)}</table>"


def report_html(results: dict[str, DayResult]) -> tuple[str, str]:
    """Subject + HTML for one day or a date range."""
    dates = sorted(results)
    bets = pd.concat([r.bets for r in results.values() if not r.bets.empty] or [pd.DataFrame()], ignore_index=True)
    first, last = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])
    period = f"{first:%d.%m.%Y}" if first == last else f"{first:%d.%m} &ndash; {last:%d.%m.%Y}"
    parts = [f"<h2>Value bets tenis Superbet vs Pinnacle &mdash; {period}</h2>",
             f"<p>Meciuri comparate: {sum(r.compared for r in results.values())} &middot; "
             f"nepotrivite pe Pinnacle: {sum(len(r.unmatched) for r in results.values())} &middot; "
             f"pariuri cu valoare: <b>{len(bets)}</b>.</p>"]
    if len(dates) > 1:
        rows = "".join(f"<tr><td>{WEEKDAY[pd.Timestamp(d).weekday()]} {pd.Timestamp(d):%d.%m}</td>"
                       f"<td>{results[d].compared}</td><td>{len(results[d].unmatched)}</td>"
                       f"<td>{len(results[d].bets)}</td></tr>" for d in dates)
        parts.append("<table border='1' cellpadding='4' cellspacing='0'><tr><th>Zi</th><th>Comparate</th>"
                     f"<th>Nepotrivite pe Pinnacle</th><th>Pariuri</th></tr>{rows}</table>")
    parts.append(_bets_table(bets, len(dates) > 1) if not bets.empty else "<p><b>Niciun pariu cu valoare.</b></p>")
    parts.append(_record_html(results[dates[-1]].summary, "Bilanț de la început"))
    parts.append(FOOTER)
    span = f"{first:%d.%m.%Y}" if first == last else f"{first:%d.%m}-{last:%d.%m.%Y}"
    return f"Value bets tenis ({len(bets)}) -- {span}", "\n".join(p for p in parts if p)


def weekly_html(state_dir: str | Path, end: str) -> tuple[str, str]:
    """Subject + HTML for the 7 days before `end` (exclusive) and the all-time record."""
    state = Path(state_dir)
    end_ts = pd.Timestamp(end)
    start_ts = end_ts - pd.Timedelta(days=7)
    stats_path = state / "daily_stats.csv"
    days = pd.read_csv(stats_path, parse_dates=["date"]) if stats_path.exists() else pd.DataFrame(
        columns=["date", "compared", "unmatched", "bets"])
    week_days = days[(days["date"] >= start_ts) & (days["date"] < end_ts)]
    settled = settle_log(state)
    week = settled[(settled["date"] >= start_ts) & (settled["date"] < end_ts)] if settled is not None else None
    period = f"{start_ts:%d.%m}-{end_ts - pd.Timedelta(days=1):%d.%m.%Y}"
    n_week = 0 if week is None else len(week)
    parts = [f"<h2>Rezumat săptămânal value bets tenis &mdash; {period}</h2>",
             f"<p>Zile rulate: {len(week_days)} din 7 &middot; meciuri comparate: {int(week_days['compared'].sum())} "
             f"&middot; pariuri găsite: {n_week}.</p>"]
    if n_week:
        parts.append(_bets_table(week, True, closing=True))
        parts.append(_record_html(summarize(week), "Săptămâna"))
    parts.append(_record_html(summarize(settled), "De la început") if settled is not None
                 else "<p>Încă niciun pariu în jurnal.</p>")
    parts.append(FOOTER)
    return f"Rezumat saptamanal value bets tenis ({n_week} pariuri) -- {period}", "\n".join(p for p in parts if p)
