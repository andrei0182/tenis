"""Tennis value-bet email: Superbet match-winner odds vs Pinnacle (pinnapi.com), plus the closing-line record.

Environment:
  PINNAPI_KEY                                   pinnapi.com API key (secret)
  GMAIL_ADDRESS, GMAIL_APP_PASSWORD, EMAIL_TO   same as send_report_email.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from tenis.daily import report_html, run_range, weekly_html
from tenis.staking import StakingConfig
from tenis.value import load_superbet_tennis, superbet_frame

TOURS = ("atp", "wta", "challenger", "wta-125", "itf-m", "itf-f")


def fetch_superbet(date: dt.date, tours: tuple[str, ...] = TOURS) -> pd.DataFrame:
    """Superbet singles match-winner odds for `date` (only the Superbet API; no TennisExplorer lookups)."""
    from tenis_scraper import events, tournaments

    rows = []
    for tour in tours:
        ids = tournaments.all_tennis_tournament_ids(tours=(tour,))
        if not ids:
            continue
        for ev in events.fetch_events_for_all_tournaments(ids, date):
            m = events.parse_event(ev, tour=tour)
            if m.odds_winner.player1 and m.odds_winner.player2:
                rows.append({"tour": tour, "tournament": m.tournament, "player1": m.player1, "player2": m.player2,
                             "odds_1": m.odds_winner.player1, "odds_2": m.odds_winner.player2})
    return superbet_frame(rows, date.isoformat())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (first day)")
    parser.add_argument("--end", default=None, help="Last day of a multi-day run (default: --date)")
    parser.add_argument("--superbet-xlsx", default=None, help="Use this tenis.xlsx instead of calling Superbet (one day)")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--ev-min", type=float, default=0.02)
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--weekly", action="store_true", help="Weekly summary of the 7 days before --date")
    parser.add_argument("--dry-run", action="store_true", help="Print the email instead of sending it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.weekly:
        subject, body = weekly_html(args.state_dir, args.date)
    else:
        dates = [d.date() for d in pd.date_range(args.date, args.end or args.date)]
        if args.superbet_xlsx:
            superbet = {args.date: load_superbet_tennis(args.superbet_xlsx, args.date)}
        else:
            superbet = {d.isoformat(): fetch_superbet(d) for d in dates}
        results = run_range(superbet, args.state_dir, StakingConfig(ev_min=args.ev_min, bankroll=args.bankroll),
                            args.ev_min)
        for d, r in results.items():
            print(f"{d}: {r.compared} comparate, {len(r.unmatched)} nepotrivite, {len(r.bets)} pariuri")
        subject, body = report_html(results)
    print(subject)
    if args.dry_run:
        print(body)
        return
    from send_report_email import send_email

    send_email(subject, body)


if __name__ == "__main__":
    main()
