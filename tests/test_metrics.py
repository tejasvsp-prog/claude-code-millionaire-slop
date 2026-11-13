import math

import pytest

from src.metrics import accuracy, brier_score, log_loss


def test_coin_flip_baselines():
    y = [1, 0, 1, 0]
    p = [0.5] * 4
    assert log_loss(y, p) == pytest.approx(math.log(2))
    assert brier_score(y, p) == pytest.approx(0.25)


def test_known_values():
    y = [1, 0]
    p = [0.8, 0.3]
    assert log_loss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert brier_score(y, p) == pytest.approx((0.2**2 + 0.3**2) / 2)
    assert accuracy(y, p) == 1.0


def test_perfect_and_confidently_wrong():
    assert log_loss([1, 0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-12)
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert math.isfinite(log_loss([1], [0.0]))  # clipped, not inf
    assert brier_score([1], [0.0]) == 1.0


@pytest.mark.parametrize(
    "y, p",
    [([1, 0], [0.5]), ([], []), ([2], [0.5]), ([1], [1.2]), ([1], [float("nan")])],
)
def test_bad_input(y, p):
    with pytest.raises(ValueError):
        log_loss(y, p)
