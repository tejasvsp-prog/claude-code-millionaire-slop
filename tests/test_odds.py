import math

import pytest

from src.odds import (
    american_to_decimal,
    american_to_implied_prob,
    bet_profit,
    decimal_to_american,
    expected_value,
    kelly_fraction,
    no_vig_probs_american,
    overround,
    prob_to_american,
    remove_vig,
)


@pytest.mark.parametrize(
    "american, decimal",
    [(-110, 1 + 100 / 110), (100, 2.0), (-100, 2.0), (150, 2.5), (-200, 1.5), (300, 4.0)],
)
def test_american_to_decimal(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal)


@pytest.mark.parametrize("american", [-110, -200, 100, 150, 300, -450])
def test_american_decimal_roundtrip(american):
    assert decimal_to_american(american_to_decimal(american)) == pytest.approx(
        100 if american == -100 else american
    )


@pytest.mark.parametrize("bad", [0, 50, -99, float("nan"), float("inf")])
def test_invalid_american_odds_raise(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


@pytest.mark.parametrize("bad", [1.0, 0.5, -2.0])
def test_invalid_decimal_odds_raise(bad):
    with pytest.raises(ValueError):
        decimal_to_american(bad)


def test_implied_prob_known_values():
    assert american_to_implied_prob(-110) == pytest.approx(110 / 210)
    assert american_to_implied_prob(100) == pytest.approx(0.5)
    assert american_to_implied_prob(-200) == pytest.approx(2 / 3)
    assert american_to_implied_prob(200) == pytest.approx(1 / 3)


def test_prob_to_american_roundtrip():
    for p in [0.2, 0.4, 0.5, 0.6, 0.8]:
        assert american_to_implied_prob(prob_to_american(p)) == pytest.approx(p)
    with pytest.raises(ValueError):
        prob_to_american(0.0)


def test_overround_standard_line():
    probs = [american_to_implied_prob(-110), american_to_implied_prob(-110)]
    assert overround(probs) == pytest.approx(220 / 210 - 1)  # ~4.76%


def test_remove_vig_sums_to_one_and_preserves_ratio():
    raw = [0.55, 0.50]
    fair = remove_vig(raw)
    assert sum(fair) == pytest.approx(1.0)
    assert fair[0] / fair[1] == pytest.approx(raw[0] / raw[1])


def test_remove_vig_rejects_bad_input():
    with pytest.raises(ValueError):
        remove_vig([0.5])
    with pytest.raises(ValueError):
        remove_vig([0.0, 1.0])


def test_no_vig_symmetric_line_is_fifty_fifty():
    home, away = no_vig_probs_american(-110, -110)
    assert home == pytest.approx(0.5)
    assert away == pytest.approx(0.5)


def test_no_vig_favorite():
    home, away = no_vig_probs_american(-200, 170)
    assert home + away == pytest.approx(1.0)
    raw_h, raw_a = 2 / 3, 100 / 270
    assert home == pytest.approx(raw_h / (raw_h + raw_a))
    assert home > 0.5 > away


def test_expected_value():
    # Fair coin at even money: zero EV.
    assert expected_value(0.5, 100) == pytest.approx(0.0)
    # Fair coin at -110: negative EV equal to the house edge.
    assert expected_value(0.5, -110) == pytest.approx(0.5 * 100 / 110 - 0.5)
    # Market-implied prob gives exactly zero EV.
    assert expected_value(american_to_implied_prob(150), 150) == pytest.approx(0.0)


def test_bet_profit():
    assert bet_profit(True, -110, stake=110) == pytest.approx(100)
    assert bet_profit(True, 150, stake=100) == pytest.approx(150)
    assert bet_profit(False, 150, stake=100) == -100
    with pytest.raises(ValueError):
        bet_profit(True, 100, stake=-1)


def test_kelly_fraction():
    # 60% at even money -> f* = 0.2
    assert kelly_fraction(0.6, 100) == pytest.approx(0.2)
    # No edge -> no bet
    assert kelly_fraction(0.5, -110) == 0.0
    assert kelly_fraction(american_to_implied_prob(200), 200) == pytest.approx(0.0, abs=1e-12)
    assert not math.isnan(kelly_fraction(0.0, 100))
