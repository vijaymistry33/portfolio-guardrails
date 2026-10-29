"""Market data: price and FX snapshots, and base-currency return series.

Conventions
-----------
* ``prices``  - local-currency closes; columns are fund tickers and index tickers.
* ``fx``      - value of one unit of each currency in the base currency
                (column ``USD`` = SGD per USD when the base is SGD). The base
                currency itself is always 1.0 and need not be supplied.
* ``factors`` - daily factor returns (not levels).

Fail-closed rule: a missing, non-positive or stale price never becomes a
silently carried-forward number. It raises ``DataQualityError`` and the
monitoring run refuses to propose trades on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Instrument


class DataQualityError(RuntimeError):
    """Prices or FX rates are missing, non-positive or stale."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    fx: pd.DataFrame
    factors: pd.DataFrame

    @classmethod
    def from_csv(cls, prices: str, fx: str, factors: str) -> MarketData:
        read = lambda p: pd.read_csv(p, index_col=0, parse_dates=True).sort_index()  # noqa: E731
        return cls(prices=read(prices), fx=read(fx), factors=read(factors))


def simple_returns(levels: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Close-to-close returns; the first row (no prior close) is dropped."""
    return (levels / levels.shift(1) - 1).iloc[1:]


def _last_good(series: pd.Series, as_of: pd.Timestamp) -> tuple[pd.Timestamp | None, float]:
    s = series.loc[:as_of].dropna()
    s = s[s > 0]
    if s.empty:
        return None, float("nan")
    return s.index[-1], float(s.iloc[-1])


def snapshot(
    md: MarketData,
    instruments: dict[str, Instrument],
    tickers: list[str],
    base_currency: str,
    as_of: pd.Timestamp,
    stale_after_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Latest local price and base-per-local FX rate for each ticker.

    Raises DataQualityError listing *every* missing or stale input.
    Staleness is counted in weekdays; exchange-specific holiday calendars
    (SGX, HKEX, ASX, NYSE differ) are a known simplification.
    """
    problems: list[str] = []
    prices, fx_rates = {}, {}
    for t in tickers:
        if t not in md.prices.columns:
            problems.append(f"{t}: no price history")
            continue
        when, px = _last_good(md.prices[t], as_of)
        if when is None:
            problems.append(f"{t}: no valid price on or before {as_of.date()}")
            continue
        lag = int(np.busday_count(when.date(), as_of.date()))
        if lag > stale_after_days:
            problems.append(f"{t}: last price {when.date()} is {lag} business days old")
        prices[t] = px

        ccy = instruments[t].currency
        if ccy == base_currency:
            fx_rates[t] = 1.0
        elif ccy not in md.fx.columns:
            problems.append(f"{t}: no FX series for {ccy}")
        else:
            fx_when, rate = _last_good(md.fx[ccy], as_of)
            if fx_when is None:
                problems.append(f"{ccy}: no valid FX rate on or before {as_of.date()}")
            elif int(np.busday_count(fx_when.date(), as_of.date())) > stale_after_days:
                problems.append(f"{ccy}: FX rate {fx_when.date()} is stale")
            else:
                fx_rates[t] = rate
    if problems:
        raise DataQualityError(problems)
    return pd.Series(prices, dtype=float), pd.Series(fx_rates, dtype=float)


def base_currency_returns(
    md: MarketData,
    instruments: dict[str, Instrument],
    tickers: list[str],
    base_currency: str,
    end: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    """Daily returns as a base-currency investor experiences them.

    Unhedged:  (1 + r_local) * (1 + r_fx) - 1
    Hedged:    r_local (hedge carry - the interest-rate differential - is ignored)
    """
    cols = {}
    for t in tickers:
        local = simple_returns(md.prices[t].loc[:end])
        ccy = instruments[t].currency
        if instruments[t].hedged or ccy == base_currency:
            cols[t] = local
        else:
            fx = simple_returns(md.fx[ccy].loc[:end])
            cols[t] = (1 + local) * (1 + fx) - 1
    out = pd.DataFrame(cols).dropna()
    if len(out) < lookback:
        raise DataQualityError([f"only {len(out)} complete return rows, need {lookback}"])
    return out.iloc[-lookback:]
