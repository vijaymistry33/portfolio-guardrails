"""Run the whole monitoring pipeline for one account on one date."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import checks
from .config import CASH, PortfolioConfig
from .diligence import FundReview, SwitchCase, compare_switch, review_fund
from .factors import FactorFit, fit_factor_model, variance_decomposition
from .market import DataQualityError, MarketData, base_currency_returns, snapshot
from .narrator import Narrative, TemplateNarrator
from .rebalance import RebalancePlan, plan_rebalance
from .risk import ewma_covariance, parametric_var, portfolio_vol, risk_contributions, tracking_error
from .timing import Calendar, OrderTiming, next_weekday, schedule_orders

RISK_LOOKBACK = 252  # EWMA covariance input
FACTOR_LOOKBACK = 756  # three years for factor betas


@dataclass
class MonitoringResult:
    config: PortfolioConfig
    as_of: pd.Timestamp
    findings: list[checks.Finding]
    narrative: Narrative | None = None
    plan: RebalancePlan | None = None
    vol: float = float("nan")
    vol_after: float = float("nan")
    drift_te: float = float("nan")
    var_95: float = float("nan")
    contributions: pd.DataFrame | None = None
    portfolio_fit: FactorFit | None = None
    decomposition: pd.Series | None = None
    fund_fits: dict[str, FactorFit] = field(default_factory=dict)
    reviews: list[FundReview] = field(default_factory=list)
    switches: list[SwitchCase] = field(default_factory=list)
    trade_date: date | None = None
    timings: list[OrderTiming] = field(default_factory=list)
    calendar_unavailable: str | None = None

    @property
    def blocked(self) -> bool:
        """True when bad data stopped the run: nothing here may be traded on."""
        return any(f.severity == "CRITICAL" for f in self.findings)


def run_monitoring(
    config: PortfolioConfig,
    md: MarketData,
    units: pd.Series,
    cash: float,
    as_of: pd.Timestamp,
    calendar: Calendar | None = None,
    trade_date: date | None = None,
    narrator=None,
) -> MonitoringResult:
    """``trade_date`` defaults to the next weekday after ``as_of``. With no
    calendar, every non-urgent order is held: no calendar is not a quiet day."""
    unknown = set(units.index) - set(config.instruments)
    if unknown:
        raise KeyError(f"holdings not in config: {sorted(unknown)}")
    tickers = sorted(set(units.index) | set(config.risky_targets))
    base = config.base_currency

    try:
        prices, fx = snapshot(md, config.instruments, tickers, base, as_of, config.stale_after_days)
        risk_rets = base_currency_returns(md, config.instruments, tickers, base, as_of, RISK_LOOKBACK)
        factor_rets = base_currency_returns(md, config.instruments, tickers, base, as_of, FACTOR_LOOKBACK)
    except DataQualityError as exc:  # fail closed: report the problem, propose nothing
        return MonitoringResult(config, as_of, checks.ranked(checks.data_findings(exc.problems)))

    res = MonitoringResult(config, as_of, findings=[])
    res.plan = plan = plan_rebalance(units, cash, prices, fx, config.instruments, config.policy)
    res.trade_date = trade_date or next_weekday(as_of.date())
    calendar = calendar or Calendar.missing("no event calendar supplied")
    res.calendar_unavailable = calendar.unavailable
    res.timings = schedule_orders(plan, config, calendar, res.trade_date)

    cov = ewma_covariance(risk_rets)
    risky_now = plan.weights_before.drop(CASH)
    res.vol = portfolio_vol(risky_now, cov)
    res.vol_after = portfolio_vol(plan.weights_after.drop(CASH), cov)
    res.contributions = risk_contributions(risky_now, cov)
    res.drift_te = tracking_error(risky_now, pd.Series(config.policy.targets).drop(CASH), cov)
    res.var_95 = parametric_var(res.vol, plan.nav)

    # Factor betas of today's holdings, applied to three years of history.
    factors = md.factors.loc[factor_rets.index[0] : as_of]
    backcast = factor_rets[risky_now.index] @ risky_now
    res.portfolio_fit = fit_factor_model(backcast, factors)
    res.decomposition = variance_decomposition(res.portfolio_fit)
    res.fund_fits = {t: fit_factor_model(factor_rets[t], factors) for t in tickers}

    reviewed = {t for t in tickers if config.instruments[t].benchmark}
    for current, candidates in config.alternatives.items():
        if current in reviewed:
            reviewed.update(candidates)
    by_ticker = {t: review_fund(config.instruments[t], md.prices, as_of, config.diligence) for t in sorted(reviewed)}
    res.reviews = [by_ticker[t] for t in sorted(reviewed) if t in tickers]
    res.switches = [
        compare_switch(by_ticker[cur], by_ticker[cand], config.instruments, config.diligence)
        for cur, cands in config.alternatives.items()
        if cur in tickers
        for cand in cands
    ]

    res.findings = checks.ranked(
        checks.drift_findings(plan, config)
        + checks.risk_findings(res.vol, res.drift_te, config)
        + checks.concentration_findings(plan.weights_before, config)
        + checks.diligence_findings(res.reviews, res.switches)
        + checks.timing_findings(res.timings, calendar, config.base_timezone)
    )
    res.narrative = (narrator or TemplateNarrator()).narrate(narrative_facts(res))
    return res


def narrative_facts(res: MonitoringResult) -> dict:
    """The only numbers a narrator may use."""
    plan, cfg = res.plan, res.config
    top = res.contributions["share"].idxmax()
    held = [t.trade.ticker for t in res.timings if t.verdict == "HOLD"]
    return {
        "portfolio": cfg.name,
        "as_of": str(res.as_of.date()),
        "currency": cfg.base_currency,
        "nav": round(plan.nav, 2),
        "breaches": plan.breaches_before,
        "rebalance_method": plan.method,
        "n_trades": len(plan.trades),
        "turnover": round(plan.turnover, 6),
        "trade_cost": round(plan.total_cost, 2),
        "residual_breaches": plan.breaches_after,
        "n_breaches": len(plan.breaches_before),
        "trade_date": str(res.trade_date),
        "held_orders": held,
        "n_held": len(held),
        "calendar_unavailable": res.calendar_unavailable,
        "vol": round(res.vol, 6),
        "vol_after_trades": round(res.vol_after, 6),
        "vol_low": cfg.policy.vol_range[0],
        "vol_high": cfg.policy.vol_range[1],
        "drift_tracking_error": round(res.drift_te, 6),
        "var_95_1d": round(res.var_95, 2),
        "var_confidence": 0.95,
        "top_risk": top,
        "top_risk_share": round(float(res.contributions.loc[top, "share"]), 6),
        "factor_betas": {k: round(float(v), 2) for k, v in res.portfolio_fit.betas.items()},
        "fund_flags": [r.ticker for r in res.reviews if r.flags],
        "switches": [f"{s.current} -> {s.candidate}: {s.reason}" for s in res.switches if s.recommend],
    }
