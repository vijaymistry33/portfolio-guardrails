"""Band-triggered rebalancing that respects board lots, minimum orders and costs.

Policy
------
1. **Trigger.** Nothing trades while every holding (and cash) sits inside its
   tolerance band. Bands exist because trading on every small drift costs
   more than the drift.
2. **Cash-flow first.** When triggered, first try a plan that sells only the
   *breached* overweights and deploys spare cash into underweights. A deposit
   is the cheapest rebalancing tool there is: it fixes drift without selling.
3. **Fall back to full.** If the cash-flow plan leaves more holdings outside
   their bands than a full rebalance-to-target would, use the full plan.
   That covers withdrawals and drift with no cash to fund it.
4. **Implementation.** Orders are rounded *down* to whole board lots (never
   oversell, never overspend), orders below the minimum size are dropped, and
   each order pays commission plus half the spread. Breached underweights are
   funded before in-band underweights. Cash can never go negative.

Whatever the plan cannot fix (usually because one board lot is a large slice
of a small account) is reported as a residual breach, not hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import CASH, Instrument, Policy

_EPS = 1e-9


@dataclass(frozen=True)
class Trade:
    ticker: str
    side: str  # BUY or SELL
    units: float
    price_local: float
    currency: str
    value_base: float  # gross, always positive
    cost_base: float


@dataclass(frozen=True)
class RebalancePlan:
    triggered: bool
    method: str  # "none", "cash-flow first" or "full"
    breaches_before: list[str]
    breaches_after: list[str]
    trades: list[Trade]
    nav: float
    weights_before: pd.Series  # includes CASH
    weights_after: pd.Series
    units_after: pd.Series
    cash_after: float

    @property
    def traded_value(self) -> float:
        return sum(t.value_base for t in self.trades)

    @property
    def total_cost(self) -> float:
        return sum(t.cost_base for t in self.trades)

    @property
    def turnover(self) -> float:
        """Gross traded value as a fraction of NAV."""
        return self.traded_value / self.nav if self.nav else 0.0


def current_weights(units: pd.Series, cash: float, px_base: pd.Series) -> tuple[pd.Series, float]:
    """Weights of each holding and of cash, and the NAV, in base currency."""
    values = units.mul(px_base.reindex(units.index))
    if values.isna().any():
        raise KeyError(f"no price for {list(values[values.isna()].index)}")
    nav = float(values.sum()) + cash
    if nav <= 0:
        raise ValueError("NAV must be positive")
    w = values / nav
    w[CASH] = cash / nav
    return w, nav


def out_of_band(weights: pd.Series, policy: Policy) -> list[str]:
    """Holdings whose weight is outside target +/- band.

    A holding with no target (e.g. a fund being phased out) has a target and
    band of zero, so any position in it counts as a breach.
    """
    breached = []
    for name in weights.index.union(pd.Index(list(policy.targets))):
        w = float(weights.get(name, 0.0))
        target, band = policy.targets.get(name, 0.0), policy.bands.get(name, 0.0)
        if abs(w - target) > band + _EPS:
            breached.append(name)
    return sorted(breached)


def plan_rebalance(
    units: pd.Series,
    cash: float,
    prices_local: pd.Series,
    fx_rates: pd.Series,
    instruments: dict[str, Instrument],
    policy: Policy,
) -> RebalancePlan:
    px_base = prices_local * fx_rates
    weights, nav = current_weights(units, cash, px_base)
    breaches = out_of_band(weights, policy)
    if not breaches:
        return RebalancePlan(False, "none", [], [], [], nav, weights, weights, units, cash)

    full = _build(units, cash, prices_local, fx_rates, instruments, policy, nav, breaches, sell_in_band=True)
    if policy.prefer_cashflows:
        lean = _build(units, cash, prices_local, fx_rates, instruments, policy, nav, breaches, sell_in_band=False)
        if len(lean[3]) <= len(full[3]):
            return _plan("cash-flow first", lean, nav, weights, breaches)
    return _plan("full", full, nav, weights, breaches)


def _plan(method, built, nav, weights, breaches) -> RebalancePlan:
    trades, units_after, cash_after, residual, weights_after = built
    return RebalancePlan(True, method, breaches, residual, trades, nav, weights, weights_after, units_after, cash_after)


def whole_lots(value_base: float, px_base: float, lot: int) -> float:
    return math.floor(value_base / px_base / lot + _EPS) * lot


def _order_allowed(inst: Instrument, units: float, px_local: float, px_base: float, policy: Policy) -> bool:
    return units > 0 and units * px_base >= policy.min_trade_base and units * px_local >= inst.min_trade_local


def _build(units, cash, prices_local, fx_rates, instruments, policy, nav, breaches, sell_in_band):
    px_base = prices_local * fx_rates
    names = sorted(set(units.index) | (set(policy.targets) - {CASH}))
    held = units.reindex(names, fill_value=0.0)
    value = held * px_base.reindex(names)
    gap = pd.Series({n: policy.targets.get(n, 0.0) * nav for n in names}) - value  # + buy / - sell

    trades: list[Trade] = []
    new_units = held.copy()
    cash_now = cash

    # 1) Sells: breached overweights always; in-band overweights only in the full plan.
    for n in names:
        if gap[n] >= 0 or (n not in breaches and not sell_in_band):
            continue
        inst = instruments[n]
        qty = min(whole_lots(-gap[n], px_base[n], inst.lot_size), held[n])
        if policy.targets.get(n, 0.0) == 0:  # exiting a fund: sell all of it, odd lot included
            qty = held[n]
        if not _order_allowed(inst, qty, prices_local[n], px_base[n], policy):
            continue
        gross = qty * px_base[n]
        cost = gross * inst.one_way_cost_bps / 1e4
        trades.append(Trade(n, "SELL", qty, prices_local[n], inst.currency, gross, cost))
        new_units[n] -= qty
        cash_now += gross - cost

    # 2) Buys: spare cash above the cash target, breached underweights funded first.
    budget = cash_now - policy.targets.get(CASH, 0.0) * nav
    wants = {n: gap[n] for n in names if gap[n] > 0}
    for group in ([n for n in wants if n in breaches], [n for n in wants if n not in breaches]):
        need = sum(wants[n] for n in group)
        if need <= 0 or budget <= 0:
            continue
        scale = min(1.0, budget / need)
        for n in sorted(group, key=lambda m: -wants[m]):
            inst = instruments[n]
            allot = min(wants[n] * scale, budget)
            qty = whole_lots(allot / (1 + inst.one_way_cost_bps / 1e4), px_base[n], inst.lot_size)
            if not _order_allowed(inst, qty, prices_local[n], px_base[n], policy):
                continue
            gross = qty * px_base[n]
            cost = gross * inst.one_way_cost_bps / 1e4
            trades.append(Trade(n, "BUY", qty, prices_local[n], inst.currency, gross, cost))
            new_units[n] += qty
            cash_now -= gross + cost
            budget -= gross + cost

    if cash_now < -1e-6 or (new_units < -_EPS).any():  # invariant, checked in production too
        raise RuntimeError("rebalance produced negative cash or a short position")
    new_units = new_units[new_units > _EPS]
    weights_after, _ = current_weights(new_units, cash_now, px_base)
    return trades, new_units, cash_now, out_of_band(weights_after, policy), weights_after
