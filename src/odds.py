"""Odds conversion and vig-removal utilities.

All probabilities are in [0, 1]. American odds follow the usual convention:
negative = favorite (amount risked to win 100), positive = underdog
(amount won on a 100 stake). American odds in (-100, 100) are invalid.
"""

from __future__ import annotations

import math
from typing import Sequence


def _check_american(odds: float) -> None:
    if not math.isfinite(odds) or -100 < odds < 100:
        raise ValueError(f"Invalid American odds: {odds!r} (must be <= -100 or >= +100)")


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds (total return per 1 unit staked)."""
    _check_american(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / -odds


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds!r}")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def american_to_implied_prob(odds: float) -> float:
    """Implied probability of American odds, *including* the bookmaker's vig."""
    return 1.0 / american_to_decimal(odds)


def prob_to_american(prob: float) -> float:
    """Fair American odds for a probability (no vig)."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {prob!r}")
    return decimal_to_american(1.0 / prob)


def overround(implied_probs: Sequence[float]) -> float:
    """Bookmaker margin: sum of implied probabilities minus 1."""
    return float(sum(implied_probs)) - 1.0


def remove_vig(implied_probs: Sequence[float]) -> list[float]:
    """Proportional (multiplicative) vig removal: normalize so probs sum to 1."""
    if len(implied_probs) < 2:
        raise ValueError("Need at least two outcomes to remove vig")
    if any(p <= 0 or p >= 1 for p in implied_probs):
        raise ValueError(f"Implied probabilities must be in (0, 1), got {implied_probs!r}")
    total = float(sum(implied_probs))
    return [p / total for p in implied_probs]


def no_vig_probs_american(home_odds: float, away_odds: float) -> tuple[float, float]:
    """No-vig (home, away) win probabilities from a two-way American moneyline."""
    home, away = remove_vig(
        [american_to_implied_prob(home_odds), american_to_implied_prob(away_odds)]
    )
    return home, away


def expected_value(prob: float, american_odds: float) -> float:
    """Expected profit per 1 unit staked, given true win probability `prob`."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {prob!r}")
    return prob * (american_to_decimal(american_odds) - 1.0) - (1.0 - prob)


def bet_profit(won: bool, american_odds: float, stake: float = 1.0) -> float:
    """Realized profit of a settled bet (stake lost if it loses)."""
    if stake < 0:
        raise ValueError("Stake must be non-negative")
    if won:
        return stake * (american_to_decimal(american_odds) - 1.0)
    return -stake


def kelly_fraction(prob: float, american_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll; 0 if the bet has no edge."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {prob!r}")
    b = american_to_decimal(american_odds) - 1.0
    return max(0.0, (b * prob - (1.0 - prob)) / b)
