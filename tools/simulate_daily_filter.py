"""Simulare istorică a raportului zilnic: estimarea compusă din main.py + filtrele din send_report_email.py
pe meciurile tennis-data.co.uk, cu cota Bet365 în locul celei Superbet.

Estimarea compusă se reconstruiește cu exact formulele din main.py, din date disponibile ÎNAINTE de meci:
  - rank: WRank/LRank din tennis-data (clasamentul de la data meciului) -> _rank_probability
  - formă: ultimele --form-n meciuri ale fiecărui jucător -> _form_probability
  - rating carieră: bilanțul V/Î din sezonul curent (an calendaristic), toate suprafețele ->
    _career_rating_probability (încredere = min(meciuri/40, 1))
Diferență față de raportul real: tennis-data are doar circuitul principal ATP/WTA, deci forma și bilanțul
nu includ meciurile de Challenger/ITF ale jucătorilor, iar filtrul pe cota de total game-uri lipsește.

    python tools/simulate_daily_filter.py data/raw [--form-n 10] > outputs/simulare_raport.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import _composite_estimate, _form_probability, _rank_probability  # noqa: E402
from send_report_email import MIN_COMPOSITE_PCT, MIN_EDGE_PP, power_devig  # noqa: E402

ROUND_ORDER = {"0th Round": 0, "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
               "Round Robin": 4, "Quarterfinals": 5, "Semifinals": 6, "The Final": 7}


def load(folder: str) -> pd.DataFrame:
    frames = []
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        d = pd.read_csv(f, low_memory=False) if f.suffix.lower() == ".csv" else pd.read_excel(f)
        if "Winner" not in d.columns:
            continue
        d["tour"] = "WTA" if ("WTA" in d.columns or "Tier" in d.columns) else "ATP"
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(["Date", "Winner", "Loser", "Tournament"])
    df["ro"] = df["Round"].map(ROUND_ORDER).fillna(3)
    return df.sort_values(["Date", "ro"]).reset_index(drop=True)


def odds_pair(w, l):
    w = pd.to_numeric(w, errors="coerce"); l = pd.to_numeric(l, errors="coerce")
    ov = 1 / w + 1 / l
    good = (w > 1) & (l > 1) & ov.between(1.0, 1.2)
    return w.where(good), l.where(good)


def build(df: pd.DataFrame, form_n: int) -> pd.DataFrame:
    """Un rând pe meci, jucătorul A ales aleator; semnalele sunt calculate înainte de meci."""
    rng = np.random.default_rng(42)
    df["B365W"], df["B365L"] = odds_pair(df.get("B365W"), df.get("B365L"))
    df["PSW"], df["PSL"] = odds_pair(df.get("PSW"), df.get("PSL"))
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_n))
    season: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    out = []
    completed = df["Comment"].astype(str).str.strip().eq("Completed").to_numpy()
    for i, r in enumerate(df.itertuples(index=False)):
        key_w, key_l = (r.tour, r.Winner), (r.tour, r.Loser)
        year = r.Date.year
        if not pd.isna(r.B365W):
            flip = rng.random() < 0.5
            a, b = (key_l, key_w) if flip else (key_w, key_l)
            rank_a, rank_b = (r.LRank, r.WRank) if flip else (r.WRank, r.LRank)
            oa, ob = (r.B365L, r.B365W) if flip else (r.B365W, r.B365L)
            pa, pb = (r.PSL, r.PSW) if flip else (r.PSW, r.PSL)
            ra = pd.to_numeric(rank_a, errors="coerce"); rb = pd.to_numeric(rank_b, errors="coerce")
            rank_p = _rank_probability(None if pd.isna(ra) else int(ra), None if pd.isna(rb) else int(rb))
            fa, fb = recent[a], recent[b]
            form_p = _form_probability(sum(fa), len(fa) - sum(fa), sum(fb), len(fb) - sum(fb))
            sa, sb = season[(a, year)], season[(b, year)]
            t1, t2 = sum(sa), sum(sb)
            if t1 and t2:
                r1, r2 = sa[0] / t1, sb[0] / t2
                rating_p = r1 / (r1 + r2) if r1 + r2 > 0 else None
                conf = min((t1 + t2) / 40, 1.0)
            else:
                rating_p, conf = None, 0.0
            comp = _composite_estimate(rank_p, form_p, rating_p, conf)
            out.append({"date": r.Date, "tour": r.tour, "won_a": int(flip is False),
                        "oa": oa, "ob": ob, "pa": pa, "pb": pb, "comp_a": comp, "conf": conf,
                        "series": str(r.Series if r.tour == "ATP" or pd.isna(getattr(r, "Tier", np.nan)) else r.Tier),
                        "completed": completed[i]})
        if completed[i]:
            recent[key_w].append(1); recent[key_l].append(0)
            season[(key_w, year)][0] += 1; season[(key_l, year)][1] += 1
    return pd.DataFrame(out)


def sides(m: pd.DataFrame) -> pd.DataFrame:
    """Ambele părți ale fiecărui meci, ca la filter_recommended_picks: partea cu edge pozitiv."""
    m = m[m.completed & m.comp_a.notna()].copy()
    impl_a = (1 / m.oa) / (1 / m.oa + 1 / m.ob)
    m["edge_a"] = (m.comp_a - impl_a) * 100
    m["weight"] = 0.3 + 0.7 * m.conf
    rows = []
    for side in ("a", "b"):
        s = pd.DataFrame({"date": m.date, "tour": m.tour, "series": m.series})
        if side == "a":
            s["comp"], s["odds"], s["odds_o"], s["ps"], s["ps_o"], s["won"], s["edge"] = (
                m.comp_a * 100, m.oa, m.ob, m.pa, m.pb, m.won_a, m.edge_a)
        else:
            s["comp"], s["odds"], s["odds_o"], s["ps"], s["ps_o"], s["won"], s["edge"] = (
                (1 - m.comp_a) * 100, m.ob, m.oa, m.pb, m.pa, 1 - m.won_a, -m.edge_a)
        s["edge_w"] = s.edge * m.weight
        rows.append(s)
    s = pd.concat(rows, ignore_index=True)
    s["market"] = [power_devig(a, b) for a, b in zip(s.odds, s.odds_o)]
    s["ps_fair"] = (1 / s.ps) / (1 / s.ps + 1 / s.ps_o)
    s["pl"] = np.where(s.won == 1, s.odds - 1, -1.0)
    s["clv"] = s.odds / (1 / s.ps_fair) - 1  # cota luată vs prețul corect Pinnacle (fără marjă)
    return s


def summary(g: pd.DataFrame, label: str) -> dict:
    n = len(g)
    if n == 0:
        return {"filtru": label, "pariuri": 0}
    se = g.pl.std(ddof=1) / np.sqrt(n) * 100 if n > 1 else np.nan
    return {"filtru": label, "pariuri": n, "cota medie": g.odds.mean(), "estimat %": g.comp.mean(),
            "castig real %": g.won.mean() * 100, "implicit %": g.market.mean() * 100,
            "profit (miza 1)": g.pl.sum(), "yield %": g.pl.mean() * 100, "± SE": se,
            "CLV vs Pinnacle %": g.clv.mean() * 100}


def fit_shrink(s: pd.DataFrame) -> tuple[float, float, float]:
    grid = np.linspace(-0.2, 1.0, 241)
    p0, c, y = s.market.to_numpy(), s.comp.to_numpy() / 100, s.won.to_numpy()
    ll = np.array([np.sum(y * np.log(np.clip(p0 + l * (c - p0), 1e-4, 1 - 1e-4))
                          + (1 - y) * np.log(np.clip(1 - p0 - l * (c - p0), 1e-4, 1 - 1e-4))) for l in grid])
    best = int(np.argmax(ll)); ok = grid[ll >= ll[best] - 1.92]
    return grid[best], ok.min(), ok.max()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--form-n", type=int, default=10)
    args = ap.parse_args()
    m = build(load(args.data), args.form_n)
    s = sides(m)
    one = s[s.edge >= 0]  # partea favorizată de estimare, un rând pe meci
    base = one[(one.edge_w >= MIN_EDGE_PP) & (one.comp >= MIN_COMPOSITE_PCT)].copy()

    print(f"Meciuri cu cote Bet365 și estimare compusă: {len(one)} | trec de filtrele vechi: {len(base)}\n")
    print("### 1. Adaugă estimarea compusă informație peste cotă?\n")
    for lab, g in [("toate meciurile", one), ("doar cele care trec de filtrele vechi", base)]:
        lam, lo, hi = fit_shrink(g)
        print(f"- {lab}: factor optim de păstrare a diferenței față de piață = **{lam:.2f}** (interval 95% {lo:.2f}–{hi:.2f})")

    rows = [summary(base, "filtre vechi (edge>=15pp, estimare>=55%)")]
    variants = []
    for shrink in (0.1, 0.2, 0.3):
        for ev, ev_long in ((0.05, 0.15), (0.0, 0.0), (0.10, 0.20)):
            p = base.market + shrink * (base.comp / 100 - base.market)
            e = p * base.odds - 1
            keep = np.where(base.odds >= 3.0, e >= ev_long, e >= ev)
            variants.append((shrink, ev, ev_long, base[keep]))
            rows.append(summary(base[keep], f"SHRINK {shrink}, EV>={ev:.0%} / >={ev_long:.0%} la cote>=3"))
    rows.append(summary(base[base.odds < 3.0], "filtre vechi, doar cote < 3.00"))
    rows.append(summary(base[base.odds >= 3.0], "filtre vechi, doar cote >= 3.00"))
    print("\n### 2. Filtrele, pe tot istoricul\n")
    print(pd.DataFrame(rows).round(1).to_markdown(index=False))

    print("\n### 3. Filtrele vechi pe grupe de cotă\n")
    base["band"] = pd.cut(base.odds, [1, 1.5, 2, 2.5, 3, 4, 6, 1000], right=False)
    print(pd.DataFrame([summary(g, str(k)) for k, g in base.groupby("band", observed=True)]).round(1).to_markdown(index=False))

    print("\n### 4. Out-of-sample: factorul estimat pe 2004–2014, testat pe 2015–2026\n")
    train, test = base[base.date.dt.year <= 2014], base[base.date.dt.year >= 2015]
    lam, lo, hi = fit_shrink(train)
    print(f"- factor pe 2004–2014: {lam:.2f} ({lo:.2f}–{hi:.2f}); pe 2015–2026 ar fi {fit_shrink(test)[0]:.2f}\n")
    rows = [summary(test, "2015–2026, filtre vechi")]
    for ev, ev_long in ((0.05, 0.15), (0.0, 0.0)):
        p = test.market + max(lam, 0) * (test.comp / 100 - test.market)
        e = p * test.odds - 1
        keep = np.where(test.odds >= 3.0, e >= ev_long, e >= ev)
        rows.append(summary(test[keep], f"2015–2026, SHRINK {max(lam, 0):.2f}, EV>={ev:.0%} / >={ev_long:.0%}"))
    print(pd.DataFrame(rows).round(1).to_markdown(index=False))

    print("\n### 5. Filtrele vechi pe circuit și perioadă\n")
    base["per"] = pd.cut(base.date.dt.year, [2003, 2009, 2014, 2019, 2030], labels=["2004-09", "2010-14", "2015-19", "2020-26"])
    rows = [summary(g, f"{t} {p}") for (t, p), g in base.groupby(["tour", "per"], observed=True)]
    print(pd.DataFrame(rows).round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
