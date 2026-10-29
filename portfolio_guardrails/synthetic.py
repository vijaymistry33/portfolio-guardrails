"""Seeded synthetic market with a *known* factor structure.

Why synthetic: the sample runs offline with no data licence, and because the
true exposures are planted, the tests can check that the estimators recover
them instead of merely checking that the code runs.

Factors (daily returns): MKT (global equity), RATES (duration), USD (USD/SGD).
Each index's local-currency return is a planted mix of factors plus noise.
Each fund returns its index, minus its fee, minus any planted leakage, plus
planted tracking noise. FX: HKD is pegged to USD (tiny noise around the peg);
AUD is a risk-on currency that partly follows the equity factor.

Every number in this file is illustrative. None is a forecast or a fact
about a real fund.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig
from .market import MarketData

FACTORS = ["MKT", "RATES", "USD"]
_VOL = np.array([0.16, 0.06, 0.05])  # annualised
_DRIFT = np.array([0.07, 0.03, 0.0])
_CORR = np.array([[1.0, -0.25, -0.15], [-0.25, 1.0, 0.05], [-0.15, 0.05, 1.0]])

# index -> (currency, local-currency betas on MKT, RATES, USD, idiosyncratic vol)
INDEX_SPECS = {
    "IDX_DM": ("USD", (1.00, 0.00, 0.0), 0.02),
    "IDX_EM": ("USD", (1.10, 0.00, 0.0), 0.10),
    "IDX_HK": ("HKD", (1.05, 0.00, 0.0), 0.15),
    "IDX_AU": ("AUD", (0.80, 0.00, 0.0), 0.09),
    "IDX_AGG": ("USD", (0.05, 1.00, 0.0), 0.01),
    "IDX_SG": ("SGD", (0.00, 0.70, 0.0), 0.02),
    "IDX_GOLD": ("USD", (0.05, 0.40, -0.6), 0.13),
}

# fund -> (planted tracking error, planted leakage beyond fee, starting price)
FUND_SPECS = {
    "GEQ": (0.0025, 0.0, 110.0),
    "EMQ": (0.0160, 0.004, 45.0),
    "EMQ_ALT": (0.0040, 0.0, 52.0),
    "HKQ": (0.0030, 0.0, 24.0),
    "AUQ": (0.0030, 0.0, 38.0),
    "IGB": (0.0030, 0.0, 48.0),
    "SGB": (0.0040, 0.0, 1.05),
    "GLD": (0.0020, 0.0, 180.0),
}


def generate_market(config: PortfolioConfig, seed: int = 7, years: int = 4, end: str = "2026-09-29") -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=years * 252 + 1)
    n = len(dates) - 1
    daily = lambda annual: annual / np.sqrt(252)  # noqa: E731

    cov = np.outer(_VOL, _VOL) * _CORR / 252
    f = rng.multivariate_normal(_DRIFT / 252, cov, size=n)
    factors = pd.DataFrame(f, index=dates[1:], columns=FACTORS)

    usd = factors["USD"].to_numpy()
    aud = 0.25 * f[:, 0] + 0.7 * usd + rng.normal(0, daily(0.06), n)
    usdsgd = 1.30 * np.cumprod(np.r_[1.0, 1 + usd])
    fx = pd.DataFrame(
        {
            "USD": usdsgd,
            "HKD": usdsgd / 7.80 * (1 + rng.normal(0, 0.0003, n + 1)),
            "AUD": 0.86 * np.cumprod(np.r_[1.0, 1 + aud]),
        },
        index=dates,
    )

    def path(start: float, r: np.ndarray) -> np.ndarray:
        return start * np.cumprod(np.r_[1.0, 1 + r])

    levels, index_r = {}, {}
    for idx, (_ccy, betas, idio) in INDEX_SPECS.items():
        index_r[idx] = f @ np.array(betas) + rng.normal(0, daily(idio), n)
        levels[idx] = path(1000.0, index_r[idx])

    for ticker, inst in config.instruments.items():
        te, leak, start = FUND_SPECS[ticker]
        r = index_r[inst.benchmark] - (inst.ter + leak) / 252 + rng.normal(0, daily(te), n)
        levels[ticker] = path(start, r)

    return MarketData(prices=pd.DataFrame(levels, index=dates), fx=fx, factors=factors)
