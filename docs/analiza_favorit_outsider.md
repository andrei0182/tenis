# Favorit vs. outsider: ce spun 80.000 de meciuri

Analiză făcută pe 2026-10-11 cu `tools/odds_band_analysis.py`, ca verificare a sfaturilor din articolul
[PariuriX – analiza unui meci de tenis](https://pariurix.com/resurse/sporturi/tenis/analiza-meci-tenis.html).

## Date

Fișierele tennis-data.co.uk (site-ul blochează descărcarea automată, așa că au fost luate din copii publice
pe GitHub, în formatul original): ATP 2004–2026 din
[gmalbert/tennis-predictions](https://github.com/gmalbert/tennis-predictions) (2020–2026) și
[0xsimulacra/MLT](https://github.com/0xsimulacra/MLT) (ATP 2000–2019, WTA 2007–2019).
Doar meciuri `Completed`, cu cote Pinnacle valide: **80.513 meciuri** (52.263 ATP, 28.250 WTA),
deci 161.026 de „pariuri” posibile (ambii jucători din fiecare meci).
Doar circuitul principal ATP/WTA — **fără Challenger și ITF**.

Coloane: *câștig real* = cât au câștigat efectiv jucătorii din grupă; *prob. Pinnacle* = probabilitatea din
cota Pinnacle după scoaterea marjei; *ROI* = randamentul unui pariu de 1 unitate pe fiecare jucător din grupă
(± = o eroare standard). Bet365 e folosit ca aproximare pentru o casă „soft” ca Superbet.

## Rezultat principal (ATP + WTA)

| Cotă Pinnacle | Pariuri | Câștig real | Prob. Pinnacle | ROI Pinnacle | ROI Bet365 | ROI cea mai bună cotă |
|---|---:|---:|---:|---:|---:|---:|
| < 1.30 | 24.351 | 85.7% | 84.1% | −0.3% ± 0.3 | −2.6% | +0.3% |
| 1.30–1.50 | 20.966 | 71.4% | 70.1% | −0.6% ± 0.4 | −3.6% | −0.4% |
| 1.50–2.00 | 37.337 | 57.2% | 57.4% | −2.9% ± 0.4 | −7.1% | −2.4% |
| 2.00–2.50 | 22.340 | 44.0% | 43.8% | −1.9% ± 0.7 | −5.7% | +1.2% |
| 2.50–3.00 | 15.408 | 35.9% | 36.0% | −2.6% ± 1.0 | −6.8% | +0.1% |
| 3.00–4.00 | 16.654 | 27.5% | 28.7% | −6.6% ± 1.2 | **−11.3%** | −1.6% |
| 4.00–6.00 | 12.380 | 19.3% | 20.6% | −8.1% ± 1.7 | **−15.2%** | −4.7% |
| > 6.00 | 11.590 | 8.6% | 10.5% | −24.2% ± 2.6 | **−34.9%** | −17.7% |

Aceeași imagine separat la ATP și WTA, în toate cele patru perioade (2004–09, 2010–14, 2015–19, 2020–26)
și pe toate suprafețele.

Pe niveluri de turneu, outsiderii cu cotă ≥ 3.00:

| Nivel | Pariuri | ROI Pinnacle | ROI Bet365 |
|---|---:|---:|---:|
| Grand Slam | 10.346 | **−20.0%** | **−27.2%** |
| 500 | 8.647 | −11.3% | −17.9% |
| 250 / International | 15.168 | −10.8% | −17.6% |
| Masters / 1000 | 6.463 | −3.4% | −11.9% |

## Concluzii

1. **Efectul favorit–outsider există și e puternic.** Favoriții (sub 1.50) câștigă puțin mai des decât spune
   cota, iar outsiderii peste 3.00 mai rar. La Pinnacle favoriții sunt aproape de zero (−0.3%), outsiderii
   pierd 7–24%. Articolul are dreptate pe acest punct, iar acum e confirmat pe un eșantion de ~500 de ori
   mai mare.
2. **La o casă „soft” (Bet365, deci probabil și Superbet), niciun grup de cote nu e profitabil pariat
   orbește.** Pe cote 3–4, punctul de plecare e **−11%**, iar peste 4 e −15% … −35%. Un model care pariază
   pe outsideri trebuie să fie mai bun cu peste 11% doar ca să iasă pe zero.
3. **Grand Slam-urile sunt cel mai rău loc pentru outsideri** (−27% la Bet365): în meciuri de 3 seturi
   câștigate favoritul își impune mai sigur nivelul.
4. **Testul din articol (outsider din afara Top 50 care bate un Top 20 la Grand Slam, apoi meciul următor):**
   ATP 377 de meciuri, ROI −13.3% ± 7.0 (la cote > 3.00: −22.4%); WTA 253 de meciuri, −3.4% ± 10.1.
   Concluzia articolului („nu pariați pe surpriză în turul următor”) se confirmă la ATP. La WTA eșantionul
   e prea mic pentru o concluzie.
5. **Cea mai bună cotă disponibilă contează mai mult decât grupa de cote.** Coloana „cea mai bună cotă”
   (maximul dintre case) e aproape de zero sau pe plus la cotele sub 3.00, unde Bet365 e la −3% … −7%.
   Diferența dintre case e mai mare decât orice avantaj realist al unui model.

## Ce înseamnă pentru proiect

- Pariurile din raportul zilnic (`stats/picks_log.csv`) au cota medie **3.12**. În grupa asta, un pariu
  la întâmplare la o casă soft pierde ~11%, iar procentul compus din raport supraestimează mult șansele
  (62% estimat față de 38% real). Profitul actual de +25 de unități pe 90 de pariuri vine din câteva cote
  mari prinse și nu e semnificativ statistic (t ≈ 1.1).
- Recomandare: penalizare sau prag mai strict pentru cote ≥ 3.00 (mai ales la Grand Slam). Accent pe
  metoda deja existentă de value betting față de Pinnacle (`value_daily.py`), care compară direct cota
  Superbet cu prețul corect.
- Limită: datele nu acoperă Challenger și ITF, unde sunt multe dintre pariurile din raport. Acolo efectul
  e de obicei la fel de mare sau mai mare (piețe mai slabe), dar nu e verificat aici.
