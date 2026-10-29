"""ETF due diligence: is each fund doing its job, and is a replacement worth the switch?

Measures (fund and index in the same currency, daily, over ``window_days``):

* tracking difference (TD) = annualised fund return - annualised index return.
* leakage = TD minus the TD a fee-only fund would have had, -TER x (1 + index
  return). Negative leakage is drag *beyond* the fee (sampling, withholding
  tax, cash drag, securities-lending shortfall). Using plain TD + TER instead
  would mistake the fee's compounding for leakage in strong years.
* tracking error (TE) = annualised standard deviation of daily return gaps.

Leakage is only treated as real when it is both material (>= 5 bps a year)
and at least two standard errors from zero. Over a few years a fund's realised
TD is noisy (its standard error is roughly TE / sqrt(years)), so flagging
every small shortfall would flag noise.

Switch analysis compares annual all-in holding cost (fee, significant leakage,
expected trading) and the one-off cost of switching (selling one, buying the
other), then reports a break-even in months. Taxes are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DiligenceLimits, Instrument
from .market import simple_returns
from .risk import TRADING_DAYS

MATERIAL_LEAKAGE = 0.0005  # 5 bps a year; below this, leakage is not worth a finding


@dataclass(frozen=True)
class FundReview:
    ticker: str
    benchmark: str
    n_obs: int
    tracking_difference: float
    tracking_error: float
    ter: float
    leakage: float
    leakage_t: float
    flags: list[str] = field(default_factory=list)

    @property
    def years(self) -> float:
        return self.n_obs / TRADING_DAYS

    @property
    def leakage_is_real(self) -> bool:
        return _leakage_is_real(self.leakage, self.leakage_t)


def _leakage_is_real(leakage: float, t: float) -> bool:
    return abs(leakage) >= MATERIAL_LEAKAGE and abs(t) >= 2


@dataclass(frozen=True)
class SwitchCase:
    current: str
    candidate: str
    current_cost: float  # annual, fraction of position
    candidate_cost: float
    annual_saving: float
    switch_cost: float  # one-off, fraction of position
    breakeven_months: float
    recommend: bool
    reason: str


def annualised(r: pd.Series) -> float:
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def review_fund(inst: Instrument, prices: pd.DataFrame, end: pd.Timestamp, limits: DiligenceLimits) -> FundReview:
    if inst.benchmark is None:
        raise ValueError(f"{inst.ticker} has no benchmark to review against")
    pair = simple_returns(prices[[inst.ticker, inst.benchmark]].loc[:end]).dropna().iloc[-limits.window_days :]
    fund, index = pair[inst.ticker], pair[inst.benchmark]

    index_annual = annualised(index)
    td = annualised(fund) - index_annual
    te = float((fund - index).std(ddof=1) * np.sqrt(TRADING_DAYS))
    years = len(pair) / TRADING_DAYS
    leakage = td + inst.ter * (1 + index_annual)
    se = te / math.sqrt(years) if years > 0 else float("inf")
    leakage_t = leakage / se if se > 0 else 0.0

    flags = []
    if len(pair) < limits.window_days:
        flags.append(f"only {len(pair)} days of history (want {limits.window_days})")
    if te > limits.max_tracking_error:
        flags.append(f"tracking error {te:.2%} above {limits.max_tracking_error:.2%}")
    if leakage < 0 and _leakage_is_real(leakage, leakage_t):
        flags.append(f"drag beyond fee of {-leakage:.2%} a year (t = {leakage_t:.1f})")
    if inst.aum_usd_m < limits.min_aum_usd_m:
        flags.append(f"AUM US${inst.aum_usd_m:,.0f}m below US${limits.min_aum_usd_m:,.0f}m (closure risk)")
    if inst.adv_usd_m < limits.min_adv_usd_m:
        flags.append(f"daily traded value US${inst.adv_usd_m:,.1f}m below US${limits.min_adv_usd_m:,.1f}m")
    if inst.spread_bps > limits.max_spread_bps:
        flags.append(f"spread {inst.spread_bps:.0f} bps above {limits.max_spread_bps:.0f} bps")
    return FundReview(inst.ticker, inst.benchmark, len(pair), td, te, inst.ter, leakage, leakage_t, flags)


def holding_cost(review: FundReview, inst: Instrument, turnover: float) -> float:
    """Annual all-in cost as a fraction of the position.

    The fee, plus leakage only when it is real (material and |t| >= 2), plus
    expected trading. Using raw realised TD instead would let one lucky or
    unlucky stretch decide a switch.
    """
    drag = review.ter - (review.leakage if review.leakage_is_real else 0.0)
    return drag + turnover * inst.one_way_cost_bps / 1e4


def compare_switch(
    current: FundReview,
    candidate: FundReview,
    instruments: dict[str, Instrument],
    limits: DiligenceLimits,
) -> SwitchCase:
    cur_i, cand_i = instruments[current.ticker], instruments[candidate.ticker]
    cur_cost = holding_cost(current, cur_i, limits.expected_turnover)
    cand_cost = holding_cost(candidate, cand_i, limits.expected_turnover)
    saving = cur_cost - cand_cost
    switch = (cur_i.one_way_cost_bps + cand_i.one_way_cost_bps) / 1e4
    months = switch / saving * 12 if saving > 0 else float("inf")

    if candidate.flags:
        recommend, reason = False, "candidate has its own flags: " + "; ".join(candidate.flags)
    elif saving <= 0:
        recommend, reason = False, "candidate is not cheaper to hold"
    elif months > limits.max_breakeven_months:
        recommend, reason = False, f"saving takes {months:.0f} months to repay the switch"
    else:
        recommend, reason = True, f"saves {saving:.2%} a year; switch repaid in {months:.1f} months"
    return SwitchCase(current.ticker, candidate.ticker, cur_cost, cand_cost, saving, switch, months, recommend, reason)
