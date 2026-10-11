import pandas as pd
import pytest

import send_report_email as sre


def test_power_devig_sums_to_one_and_shrinks_longshot():
    pa = sre.power_devig(1.30, 3.50)
    pb = sre.power_devig(3.50, 1.30)
    assert pa + pb == pytest.approx(1.0, abs=1e-9)
    prop_b = (1 / 3.50) / (1 / 1.30 + 1 / 3.50)
    assert pb < prop_b  # power gives the longshot less than proportional


def test_value_check_pulls_estimate_to_market():
    market = sre.power_devig(3.2, 1.36)
    adj, ev = sre.value_check(62.0, 3.2, 1.36, shrink=0.3)
    assert adj == pytest.approx((market + 0.3 * (0.62 - market)) * 100)
    assert ev == pytest.approx(adj / 100 * 3.2 - 1)
    assert sre.value_check(62.0, None, 1.36) is None


def test_longshots_need_more_ev():
    assert sre.passes_value(0.06, 2.0)
    assert not sre.passes_value(0.06, 3.5)
    assert sre.passes_value(0.16, 3.5)


def _row(comp1, odds1, odds2):
    return {
        sre._COL_P1: "A", sre._COL_P2: "B", sre._COL_ODDS1: odds1, sre._COL_ODDS2: odds2,
        sre._COL_COMP1: comp1, sre._COL_COMP2: 100 - comp1,
        sre._COL_IMPL1: round(100 / odds1 / (1 / odds1 + 1 / odds2), 1),
        sre._COL_IMPL2: round(100 / odds2 / (1 / odds1 + 1 / odds2), 1),
        sre._COL_RATING_CONF: 1.0, sre._COL_GAMES_LINE_P1: 10.5, sre._COL_GAMES_ODDS_P1: 1.6,
        sre._COL_GAMES_LINE_P2: 10.5, sre._COL_GAMES_ODDS_P2: 1.6,
    }


def test_filter_drops_inflated_longshot_keeps_real_value():
    df = pd.DataFrame([_row(55.0, 3.60, 1.30),   # longshot: adjusted EV ~ +12%, below the 15% longshot bar
                       _row(80.0, 2.10, 1.75)])  # adjusted ~51% at 2.10 -> EV ~ +7%
    raw = sre.filter_recommended_picks(df, value_filter=False)
    picks = sre.filter_recommended_picks(df)
    assert len(raw) == 2
    assert list(picks[sre._COL_ODDS1]) == [2.10]
    assert picks.iloc[0]["_ev"] >= sre.MIN_EV


def test_calibration_table_totals():
    log = pd.DataFrame({"rec_odds": ["2.0", "3.0", "3.5"], "comp_pct": ["60", "60", "60"],
                        "result": ["won", "lost", "lost"]})
    t = sre.calibration_table(log).set_index("cota")
    assert t.loc["TOTAL", "n"] == 3
    assert t.loc["TOTAL", "profit"] == pytest.approx(1.0 - 1 - 1)
    assert t.loc["TOTAL", "real"] == pytest.approx(100 / 3)
