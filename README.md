# Tenis

Repo-ul are două părți independente:

1. **`src/tenis/`** — model de probabilitate pentru tenis (Elo pe suprafață + Markov Klaassen-Magnus),
   blend cu piața, miză Kelly fracționat și backtest walk-forward. Descris mai jos.
2. **`tenis_scraper/` + scripturile din rădăcină** — scraperul Superbet / TennisExplorer și raportul zilnic
   (neschimbate, documentate în secțiunea [Scraper Superbet + TennisExplorer](#scraper-superbet--tennisexplorer)).

---

## Modelul de tenis (`tenis`)

### Instalare

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest            # toate testele trebuie să fie verzi
```

Python 3.11+. Dependențe: numpy, pandas, scipy, scikit-learn, openpyxl, typer (+ pytest).

### Date

`data/` și `outputs/` sunt în `.gitignore`. Nu se face scraping: fișierele se descarcă manual.

1. **Rezultate + cote**: [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php), fișierele anuale ATP/WTA
   `.xlsx` / `.xls` (sau exportate `.csv`) direct în `data/raw/` (un folder separat pentru ATP și WTA dacă vrei rapoarte separate).
   Fișierele WTA sunt recunoscute după coloana `WTA` sau după „wta” în nume.
2. **Statistici de serviciu** (doar pentru modelul Markov): CSV-urile `atp_matches_YYYY.csv` /
   `wta_matches_YYYY.csv` din [tennis_atp](https://github.com/JeffSackmann/tennis_atp) /
   [tennis_wta](https://github.com/JeffSackmann/tennis_wta) (Jeff Sackmann) în `data/raw/sackmann/`.
3. **Mapping nume** (opțional): `data/player_map.csv` cu coloanele `sackmann_name,td_name`
   (ex. `Carlos Alcaraz,Alcaraz Garfia C.`). Restul numelor se potrivesc automat
   („Novak Djokovic” → „Djokovic N.”); cele nepotrivite apar în raport și în `outputs/unmatched_players.csv`.

Coloane tennis-data folosite: `Date, Tournament, Series/Tier, Surface, Round, Best of, Winner, Loser,
WRank, LRank, Comment` + cote.

| Coloane | Rol |
|---|---|
| `AvgW, AvgL` (implicit; fallback `B365W, B365L`), `--odds-source B365` pentru invers | piața **la momentul pariului**: intră în blend și e cota la care se „plasează” pariul |
| `PSW, PSL` (Pinnacle) | piața de **referință**: doar CLV (`cota_luată / cota_Pinnacle − 1`) și baseline de log loss; niciodată input în model sau blend |

Reguli de curățare:
- meciurile cu `Comment != "Completed"` (retiruri, walkover, descalificări) **nu** actualizează ratingurile și nu intră în backtest;
  la fel meciurile Sackmann cu `RET`, `W/O`, `DEF` în scor;
- numele se normalizează la o cheie unică (`"Djokovic N."` → `djokovic n`);
- fiecare meci e orientat aleator A vs. B (seed fix, `--seed`), ca să nu existe bias „câștigătorul e primul”;
- ordinea în aceeași zi: runde mai mici întâi. Carpet e tratat ca Hard;
- perechile de cote imposibile (cotă ≤ 1 sau `1/cotă_W + 1/cotă_L` în afara intervalului [1.00, 1.20], ex. o cotă 161
  introdusă greșit în sursă) devin lipsă, separat pentru cotele de pariere și pentru Pinnacle.

### Modele

- **Elo** (`elo.py`): rating general `R_g` și pe suprafață `R_s` (Hard/Clay/Grass), start 1500,
  `R += K·(rezultat − P)`, `K = 250 / (n_meciuri + 5)^0.4`. Rating combinat `R = w·R_s + (1−w)·R_g`;
  `w ∈ {0, 0.1, …, 1}` e ales pe log loss folosind ratingurile **pre-meci** ale meciurilor dinainte de `--start`
  (după un burn-in de 365 de zile). `P(A bate B) = 1 / (1 + 10^((R_B − R_A)/400))`.
  Pentru meciurile best-of-5 probabilitatea (considerată best-of-3) se convertește prin probabilitatea pe set:
  `s` din `s²(3−2s) = P`, apoi `P5 = s³(1 + 3q + 6q²)` (favoritul devine mai favorit).
- **Markov Klaassen-Magnus** (`markov.py`): `p_A = f_t + (f_A − f_medie) − (g_B − g_medie)`, unde `f` = % puncte
  câștigate la serviciu, `g` = % câștigate la retur (decădere exponențială în timp, `ξ = 0.002/zi`, shrinkage
  spre media turului), `f_t` = media de serviciu pe suprafață (aproximare pentru turneu).
  Ierarhie exactă punct → game (formula închisă cu deuce) → tiebreak (DP cu alternarea serviciului) →
  set (DP pe game-uri) → meci (best-of-3/5, serverul primului set din fiecare set urmărit exact, primul serviciu
  al meciului mediat 50/50). Piețe derivate: distribuția scorului pe seturi, handicap seturi, total game-uri.
  **Anti-leakage**: Sackmann are doar data de început a turneului, deci un meci Sackmann devine disponibil
  abia la `tourney_date + 15 zile` (turneul sigur terminat).
- **Blend** (`blend.py`): de-vig proporțional (sau `--devig-method power`), apoi regresie logistică pe
  `[logit(P_model), logit(P_piață)]`. Se reantrenează la fiecare `--refit-days` (30) zile **doar pe meciuri deja
  terminate** (minim 300).
- **Value & miză** (`staking.py`): `EV = P_final·cotă − 1`, pariu pe jucătorul cu EV maxim dacă `EV ≥ --ev-min`
  (0.03). `f* = (P·cotă − 1)/(cotă − 1)`, `miză = --kelly (0.25) · f* · bankroll`, plafon `--cap` (2%) din bankroll.
  Mizele unei zile se calculează pe bankroll-ul de la începutul zilei.

### Backtest walk-forward

- Ratingurile Elo se calculează secvențial: fiecare meci primește ratingurile de **dinainte** de el,
  iar ratingul se actualizează abia după ce meciul e „terminat” în ordinea cronologică.
- Statisticile Markov folosesc doar turnee terminate (lag de 15 zile).
- Blenderul pentru fiecare bloc de 30 de zile e antrenat doar pe meciurile dinaintea blocului.
- Pariurile și toate metricile se calculează doar pe meciurile cu dată `≥ --start`.
- Filtrele `--surface` și `--series` restrâng doar evaluarea și pariurile (ratingurile folosesc toate meciurile).

Raport (`outputs/summary.json`, `comparison.csv`, consolă):
- tabel **Elo vs. Markov vs. blend Elo vs. blend Markov vs. piață (Avg) vs. Pinnacle**: log loss, Brier,
  nr. pariuri, yield, ROI pe bankroll, CLV mediu — fiecare sursă (inclusiv Pinnacle, care lipsește la o parte din
  meciuri) comparată cu piața pe aceleași meciuri;
- pentru strategia principală (`blend_elo`, sau `--model markov`): nr. pariuri, hit rate, yield, ROI, max drawdown, CLV;
- rapoarte separate pe suprafață (Hard/Clay/Grass) și pe nivel de turneu (Grand Slam, Masters, ATP 500, ATP 250;
  WTA 1000/500/250 și vechile Premier/International sunt mapate pe aceleași niveluri);
- verdict explicit pentru fiecare model: `ATENȚIE: ... NU bate piața la log loss walk-forward -> nu are edge demonstrat`.

Fișiere: `bets.csv` (toate pariurile), `predictions.csv`, `comparison.csv`, `summary.json`,
`unmatched_players.csv` (dacă există nume Sackmann nepotrivite).

### Comenzi CLI

```bash
# ratinguri pe toate meciurile + blendere (+ statistici Markov dacă există data/raw/sackmann/)
tenis fit --data data/raw --out outputs/elo.pkl

# un meci; cu cote -> blend, EV și miză
tenis predict --model outputs/elo.pkl --player-a "Sinner J." --player-b "Alcaraz C." --surface Clay
tenis predict --model outputs/elo.pkl --player-a "Sinner J." --player-b "Alcaraz C." --surface Hard \
    --best-of-5 --odds-a 1.85 --odds-b 2.05 --bankroll 1000

# backtest
tenis backtest --data data/raw --start 2022-01-01 --surface clay --ev-min 0.03 --kelly 0.25
tenis backtest --data data/raw --start 2022-01-01 --series "Grand Slam" --model markov
```

Alte opțiuni: `--cap 0.02`, `--bankroll 1000`, `--refit-days 30`, `--w 0.5` (fixează ponderea suprafeței),
`--odds-source B365`, `--devig-method power`, `--sackmann <folder>`, `--player-map <csv>`, `--seed 42`,
`--out-dir outputs`. `tenis <comandă> --help` pentru detalii.

### Limitări

- Elo nu știe de accidentări, oboseală, motivație, condiții (altitudine, minge, indoor/outdoor).
- Jucătorii noi pornesc de la 1500 (supraestimează calificații slabi, subestimează juniorii puternici) până acumulează meciuri.
- Pentru `f_t` se folosește media pe suprafață, nu media turneului (numele turneelor diferă între surse).
- Lag-ul de 15 zile pentru statisticile Sackmann e conservator: se pierde forma din ultimele două săptămâni.
- Seturile sunt tratate ca independente, cu aceleași probabilități de serviciu (fără momentum).
- `AvgW/AvgL` sunt cote medii de dinainte de meci; cota reală la care pariezi poate fi alta. Fără limite de miză.

### Note importante

- Un model nu garantează profit; majoritatea pariorilor pierd pe termen lung.
  Singurul semn credibil de edge e **CLV pozitiv constant pe un eșantion mare** (sute/mii de pariuri).
- Dacă modelul nu bate piața la log loss walk-forward, nu are edge, indiferent de ROI-ul pe perioade scurte.
- Mizele din backtest sunt teoretice; în realitate casele limitează conturile câștigătoare și cotele se mișcă.
- Pariază doar sume pe care îți permiți să le pierzi.

---

## Scraper Superbet + TennisExplorer

**Status: Phase 0 — scaffold de reconnaissance, copiat din structura care a
funcționat la [[bet]] și [[SuperBet]], nu un scraper funcțional încă.**

### Scop

Pentru meciurile de tenis disponibile pe Superbet.ro, combină:
1. **Cotele Superbet** (1X2 / minim 1 set / total game-uri) pentru meciurile
   zilei
2. **Statistici jucători** de pe Tennis Explorer (H2H, formă recentă, record
   pe suprafață) — plus orice semnal de accidentare/retragere disponibil

...într-un raport Excel zilnic care arată șansa estimată vs. cota oferită.

### De ce pornim tot ca scaffold, nu cod gata făcut

Exact lecția de la SuperBet: presupunerile greșite despre structura datelor
(JSON API vs. randare JS vs. HTML server-side) au consumat timp real la
`bet` și `SuperBet`. Aici avem DOUĂ surse necunoscute, nu una:

- **Superbet.ro — tenis**: structura URL e CONFIRMATĂ (vezi mai jos), dar
  endpoint-ul JSON de events găsit la fotbal (`sports=5`) — sport id-ul
  pentru tenis NU e confirmat, și numele market-ului pentru "meci câștigat"
  (nu există egalitate la tenis, deci nu va fi "Final" cu 1/X/2 — probabil
  doar 1/2, dar numele exact al market-ului trebuie verificat cu
  `tools/inspect_page.py` sau curl, ca la fotbal).
- **TennisExplorer.com**: NU știm încă dacă folosește același tipar AJAX cu
  token `ts=` ca BetExplorer (sunt site-uri surori, dar tenisul are pagini
  de jucător, nu pagini de echipă/clasament — structura poate diferi mult).
  Trebuie investigat separat, cu `tools/inspect_page.py`.

**Nu presupune, investighează întâi.**

### Confirmat (2026-09-15, prin browsing manual)

- Superbet.ro grupează tenisul sub `/pariuri-sportive/tenis/{tur}/{slug-turneu}/toate`,
  unde `{tur}` e unul din: `atp`, `wta`, `wta-125`, `challenger` (posibil și
  `itf`, `esport-tenis` separat, `tenis-de-masa` separat — tenis de masă NU
  e același sport).
  Exemple văzute: `tenis/atp/atp-us-open`, `tenis/wta/roland-garros-2025`,
  `tenis/challenger/biella-ita`, `tenis/wta-125/valencia-spa`.
  Asta diferă de fotbal, unde slug-ul e `fotbal---{țară}---{ligă}` — la
  tenis pare să fie `tenis---{tur}---{turneu}` (de confirmat exact în
  `sportTournamentMap_ro-RO.json`, care conține și intrările de tenis, nu
  doar fotbal — `tournaments.py` de la SuperBet doar filtra pe `fotbal---`).
- Cotele afișate pe pagină includ: rezultat final (1/2, fără egalitate),
  minim 1 set, total game-uri, handicap game-uri, scor corect.
- Perechi de dublu apar separat (ex. "Challenger - Biella (ITA) - Dublu") —
  de exclus sau tratat separat față de simplu.

### NEconfirmat — de investigat primul (Phase 0)

1. Sport id-ul pentru tenis în endpoint-ul `v3/ro-RO/events` (la fotbal e
   `sports=5`).
2. Numele exact al market-ului de "rezultat final" pentru tenis (probabil
   NU se numește "Final" cu odds 1/X/2 ca la fotbal, ci ceva cu doar 1/2).
3. Dacă TennisExplorer.com răspunde deja la `requests` simplu, sau are
   nevoie de Selenium / are un tipar AJAX diferit.
4. Cum arată o pagină de jucător pe TennisExplorer (H2H, formă, record pe
   suprafață) — structura HTML/selectori.
5. Dacă există un semnal fiabil de accidentare/retragere (WD/RET) — cel mai
   probabil doar dispariția jucătorului din lista Superbet, nu un feed
   dedicat.

### Setup

```bash
pip install -r requirements.txt
```

### Pasul 1 — investighează (fă asta primul)

```bash
python tools/inspect_page.py "https://superbet.ro/pariuri-sportive/tenis/atp/atp-us-open/toate"
python tools/inspect_page.py "https://www.tennisexplorer.com/player/<nume-jucator>/"
```

Verifică și manual, în DevTools → Network → Fetch/XHR, pe o pagină de tenis
Superbet și pe o pagină de jucător TennisExplorer, exact cum s-a făcut la
celelalte două proiecte.

### Structura proiectului

- `tenis_scraper/driver.py` — headless Chrome, copiat identic din
  SuperBet (folosește-l doar dacă investigarea arată că e nevoie de JS).
- `tenis_scraper/models.py` — model de date pentru meci de tenis (jucători,
  nu echipe; fără egalitate; seturi/game-uri, nu goluri).
- `tenis_scraper/tournaments.py` — adaptat din SuperBet pentru slug-urile
  `tenis---{tur}---{turneu}` — NEconfirmat, de verificat cu JSON-ul real.
- `tenis_scraper/events.py` — copiat din tiparul SuperBet (JSON API), cu
  `TENNIS_SPORT_ID` marcat UNCONFIRMED și `parse_event` marcat de rescris
  pentru forma reală a market-ului de tenis.
- `tenis_scraper/tennisexplorer.py` — modul NOU, scaffold gol pentru
  H2H/formă/suprafață — nimic confirmat încă, doar structura de funcții.
- `tenis_scraper/export.py` — adaptat din SuperBet, coloane pentru tenis.
- `tools/inspect_page.py` — copiat identic, generic pentru orice URL.
- `output/` — unde vor merge fișierele Excel exportate (gitignored).

### Odată ce datele sunt confirmate

Actualizează acest README și `tenis_scraper/models.py` cu structura reală,
apoi construiește logica de extragere — exact fluxul care a funcționat la
celelalte două proiecte.
