# Simularea raportului zilnic pe istoric

Analiză din 2026-10-11, făcută cu `tools/simulate_daily_filter.py`. Datele sunt aceleași fișiere tennis-data ca în
`docs/analiza_favorit_outsider.md`.

## Metoda

Estimarea compusă a fost reconstruită cu **exact funcțiile din `main.py`** (`_rank_probability`,
`_form_probability`, `_composite_estimate`). Toate semnalele sunt calculate doar din informații disponibile
**înainte** de fiecare meci:

| Semnal | În raportul real | În simulare |
|---|---|---|
| Clasament | TennisExplorer | `WRank`/`LRank` din tennis-data (clasamentul de la data meciului) |
| Formă | ultimele meciuri de pe TennisExplorer | ultimele 10 meciuri (verificat și cu 5) |
| Rating carieră | bilanțul pe suprafețe de pe TennisExplorer | bilanțul V/Î din sezonul curent, cu aceeași încredere `min(meciuri/40, 1)` |
| Cotă Superbet | Superbet | Bet365 (tot o casă „soft”) |

Pe rezultatele simulate au fost aplicate filtrele din `send_report_email.py`: edge ponderat ≥ 15pp și estimare ≥ 55%.

Au fost 89.593 de meciuri ATP (2004–2026) și WTA (2007–2019) cu cote și estimare. Dintre ele, **2.134 trec de filtrele
actuale**, adică de ~24 de ori mai multe decât cele 90 de pariuri reale.

## Rezultate

**1. Estimarea compusă nu adaugă aproape nimic peste cotă.** Factorul optim cu care trebuie păstrată diferența dintre
estimare și piață este **0.02** (interval 95%: 0.01–0.04). Practic, probabilitatea corectă e cea din cotă. Pe meciurile
care trec de filtre factorul e chiar **−0.02**: acolo unde estimarea se abate cel mai mult de la piață, are dreptate
piața.

**2. Filtrele actuale pierd, constant:**

| | Pariuri | Cotă medie | Estimat | Câștig real | Implicit din cotă | Yield | CLV vs Pinnacle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Filtre actuale | 2.134 | 2.6 | 65.1% | 36.6% | 37.5% | **−12.7% ± 2.7** | −6.3% |
| doar cote < 3.00 | 1.603 | 2.3 | 66.6% | 41.7% | 42.1% | −9.1% ± 2.8 | −6.5% |
| doar cote ≥ 3.00 | 531 | 3.8 | 60.7% | 21.1% | 23.6% | −23.7% ± 6.7 | −5.8% |
| cu formă din ultimele 5 meciuri | 3.108 | 2.6 | 67.2% | 36.6% | 38.8% | −15.9% ± 2.1 | −6.5% |

Pierderea apare în fiecare perioadă, la ATP și la WTA: yield între −5% și −18%. Estimarea spune ~65%, câștigul real
este ~37%, iar cota spunea ~38%, exact ca pe cele 90 de pariuri reale.

**3. Niciun prag de EV nu repară metoda.** Toate combinațiile testate (factor 0.1 / 0.2 / 0.3, EV ≥ 0% / 5% / 10%,
cu prag mai mare la cote ≥ 3.00) au yield între −12% și −23% și CLV negativ. Testul out-of-sample, cu factorul estimat
pe 2004–2014 și aplicat pe 2015–2026, dă factor ≤ 0, deci zero pariuri.

**4. Cele 90 de pariuri reale (+28% ± 26%) sunt compatibile cu −13%:** diferența e de ~1.6 erori standard, deci
noroc pe un eșantion mic.

## Limite

- tennis-data conține doar circuitul principal. Forma și bilanțul nu includ meciurile de Challenger/ITF, iar multe
  pariuri reale sunt chiar la Challenger/ITF.
- Filtrul pe cota de total game-uri nu poate fi simulat.
- Bet365 ține locul Superbet. Marjele sunt apropiate, dar nu identice.

Niciuna dintre aceste diferențe nu explică un factor de ~0.02. Combinația „clasament + formă + bilanț” e
informație publică pe care piața o are deja inclusă în cote.

## Concluzie

Recomandările din raportul zilnic **nu au avantaj față de piață**. Pe istoric pierd ~13% din sumele pariate.
Singura metodă din proiect cu bază solidă este compararea directă cu prețul Pinnacle (`value_daily.py`).
