"""Favorit vs. outsider: randament pe grupe de cotă pe datele tennis-data.co.uk.

Verifică, pe zeci de mii de meciuri, dacă o grupă de cote (favoriți / outsideri) are randament
mai bun decât altele la Pinnacle (piața de referință), Bet365 (casă „soft”, ca Superbet) și Max.
Include și testul din articolul PariuriX (outsider din afara Top 50 care bate un Top 20 la Grand Slam).

    python tools/odds_band_analysis.py data/raw > outputs/analiza_cote.md

`data/raw` = orice folder cu fișierele anuale tennis-data (.xlsx/.xls/.csv), căutat recursiv;
ATP/WTA se recunoaște după coloana `WTA` / `Tier`.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

frames = []
for f in sorted(Path(sys.argv[1]).rglob("*")):
    if f.suffix.lower() not in (".csv", ".xlsx", ".xls"):
        continue
    d = pd.read_csv(f, low_memory=False) if f.suffix.lower() == ".csv" else pd.read_excel(f)
    if "Winner" not in d.columns:
        continue
    d["tour"] = "WTA" if ("WTA" in d.columns or "Tier" in d.columns) else "ATP"
    if "Tier" in d.columns:
        d["Series"] = d["Tier"]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Comment"].astype(str).str.strip() == "Completed"]
df = df.drop_duplicates(["Date", "Winner", "Loser", "Tournament"])

def ok(w, l):
    w = pd.to_numeric(w, errors="coerce"); l = pd.to_numeric(l, errors="coerce")
    ov = 1 / w + 1 / l
    good = (w > 1) & (l > 1) & ov.between(1.0, 1.2)
    return w.where(good), l.where(good)

for b in ["PS", "B365", "Max", "Avg"]:
    df[b + "W"], df[b + "L"] = ok(df.get(b + "W"), df.get(b + "L"))
df = df[df.PSW.notna()]

def series(s, tour):
    s = str(s)
    if "Grand Slam" in s: return "Grand Slam"
    if s in ("Masters", "Masters 1000", "Masters Cup", "Tier 1", "Premier Mandatory", "WTA1000", "Premier 5"): return "Masters/1000"
    if s in ("ATP500", "International Gold", "Tier 2", "Premier", "WTA500"): return "500"
    return "250/International"
df["lvl"] = [series(s, t) for s, t in zip(df.Series, df.tour)]
df["Surface"] = df.Surface.replace({"Carpet": "Hard", "Greenset": "Hard"})

# one row per player side
rows = []
for side, opp, won in [("W", "L", 1), ("L", "W", 0)]:
    r = pd.DataFrame({
        "date": df.Date, "tour": df.tour, "surf": df.Surface, "lvl": df.lvl, "round": df.Round,
        "won": won, "ps": df["PS" + side], "ps_o": df["PS" + opp],
        "b365": df["B365" + side], "mx": df["Max" + side], "avg": df["Avg" + side],
        "rank": pd.to_numeric(df[side + "Rank"], errors="coerce"),
        "orank": pd.to_numeric(df[opp + "Rank"], errors="coerce"),
    })
    rows.append(r)
X = pd.concat(rows, ignore_index=True)
X["p_fair"] = (1 / X.ps) / (1 / X.ps + 1 / X.ps_o)
X["year"] = X.date.dt.year
bands = [1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1000]
labels = ["<1.30", "1.30-1.50", "1.50-2.00", "2.00-2.50", "2.50-3.00", "3.00-4.00", "4.00-6.00", ">6.00"]
X["band"] = pd.cut(X.ps, bands, labels=labels, right=False)

def roi(won, odds):
    m = odds.notna()
    pl = np.where(won[m] == 1, odds[m] - 1, -1.0)
    if len(pl) == 0: return np.nan, np.nan, 0
    return pl.mean() * 100, pl.std(ddof=1) / np.sqrt(len(pl)) * 100, len(pl)

def table(G, by="band"):
    out = []
    for k, g in G.groupby(by, observed=True):
        rp, sp, n = roi(g.won, g.ps)
        rb, sb, nb = roi(g.won, g.b365)
        rm, sm, nm = roi(g.won, g.mx)
        out.append({by: k, "n": n, "castig_real_%": g.won.mean() * 100, "prob_Pinnacle_%": g.p_fair.mean() * 100,
                    "ROI_Pinnacle_%": rp, "±": sp, "ROI_B365_%": rb, "±b": sb, "ROI_Max_%": rm})
    return pd.DataFrame(out).round(1)

def md(t, title):
    print(f"\n### {title}\n")
    print(t.to_markdown(index=False))

print(f"Meciuri: {len(df)} | ATP {df.tour.eq('ATP').sum()} ({df[df.tour=='ATP'].Date.dt.year.min()}-{df[df.tour=='ATP'].Date.dt.year.max()}) | WTA {df.tour.eq('WTA').sum()} ({df[df.tour=='WTA'].Date.dt.year.min()}-{df[df.tour=='WTA'].Date.dt.year.max()})")
md(table(X), "Toate (ATP+WTA), pe grupe de cotă Pinnacle")
for t in ["ATP", "WTA"]:
    md(table(X[X.tour == t]), f"{t}")
X["per"] = pd.cut(X.year, [2003, 2009, 2014, 2019, 2030], labels=["2004-09", "2010-14", "2015-19", "2020-26"])
X["fav"] = np.where(X.ps < 2.0, "favorit (<2.00)", np.where(X.ps < 3.0, "outsider 2.00-3.00", "outsider >=3.00"))
for col, name in [("per", "perioadă"), ("lvl", "nivel turneu"), ("surf", "suprafață")]:
    t = X.groupby([col, "fav"], observed=True).apply(lambda g: pd.Series(dict(zip(
        ["n", "castig_real_%", "prob_Pinnacle_%", "ROI_Pinnacle_%", "ROI_B365_%"],
        [len(g), g.won.mean() * 100, g.p_fair.mean() * 100, roi(g.won, g.ps)[0], roi(g.won, g.b365)[0]])))).round(1).reset_index()
    md(t, f"Favorit vs. outsider pe {name}")

# Article test: GS, player ranked outside Top 50 beats Top 20 -> next match in same tournament
gs = df[df.lvl == "Grand Slam"].copy()
gs["WR"] = pd.to_numeric(gs.WRank, errors="coerce"); gs["LR"] = pd.to_numeric(gs.LRank, errors="coerce")
ups = gs[(gs.WR > 50) & (gs.LR <= 20)]
res = []
for _, u in ups.iterrows():
    nxt = gs[(gs.Tournament == u.Tournament) & (gs.Date.dt.year == u.Date.year) & (gs.Date > u.Date) &
             ((gs.Winner == u.Winner) | (gs.Loser == u.Winner))].sort_values("Date").head(1)
    if len(nxt) == 0: continue
    n = nxt.iloc[0]; won = int(n.Winner == u.Winner)
    odds = n.PSW if won else n.PSL
    if pd.isna(odds): continue
    res.append({"tour": u.tour, "won": won, "odds": odds})
R = pd.DataFrame(res)
print("\n### Test din articol: jucător din afara Top 50 care bate un Top 20 la Grand Slam -> meciul următor\n")
for t, g in R.groupby("tour"):
    r, s, n = roi(g.won, g.odds)
    print(f"- {t}: {n} meciuri, câștigă {g.won.mean()*100:.1f}%, cotă medie {g.odds.mean():.2f}, ROI la Pinnacle {r:.1f}% ± {s:.1f}")
    for lab, m in [("cotă <= 1.50", g.odds <= 1.5), ("cotă > 3.00", g.odds > 3.0)]:
        r2, s2, n2 = roi(g.won[m], g.odds[m]); print(f"  - {lab}: {n2} meciuri, ROI {r2:.1f}% ± {s2:.1f}")
