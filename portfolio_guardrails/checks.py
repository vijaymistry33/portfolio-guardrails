"""Governance checks: turn analytics into a ranked list of findings.

Severities, in the order a reviewer should read them:

* CRITICAL - the run cannot be trusted (bad data). No trades are proposed.
* BREACH   - a policy limit is broken. Action required today.
* WARN     - approaching a limit, or a fund needs a closer look.
* INFO     - worth knowing, no action.

Each check is a plain function returning findings, so adding one is a
two-line change and each is testable alone.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from .config import CASH, PortfolioConfig
from .diligence import FundReview, SwitchCase
from .rebalance import RebalancePlan
from .timing import Calendar, OrderTiming

SEVERITY_ORDER = {"CRITICAL": 0, "BREACH": 1, "WARN": 2, "INFO": 3}


@dataclass(frozen=True)
class Finding:
    severity: str
    check: str
    subject: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def data_findings(problems: list[str]) -> list[Finding]:
    return [Finding("CRITICAL", "data", p.split(":")[0], p) for p in problems]


def drift_findings(plan: RebalancePlan, config: PortfolioConfig) -> list[Finding]:
    out = []
    pol = config.policy
    for name in plan.breaches_before:
        w, t = float(plan.weights_before.get(name, 0.0)), pol.targets.get(name, 0.0)
        out.append(
            Finding(
                "BREACH", "drift", name, f"weight {w:.1%} vs target {t:.1%} (band +/-{pol.bands.get(name, 0.0):.1%})"
            )
        )
    for name in plan.breaches_after:
        w = float(plan.weights_after.get(name, 0.0))
        lot = config.instruments[name].lot_size if name in config.instruments else None
        why = f"; one board lot is {lot:,} units" if lot and lot > 1 else ""
        out.append(Finding("WARN", "residual drift", name, f"still {w:.1%} after the proposed trades{why}"))
    for name, w in plan.weights_before.items():
        if name in plan.breaches_before or name not in pol.targets:
            continue
        band, t = pol.bands[name], pol.targets[name]
        if abs(w - t) > 0.75 * band:
            out.append(Finding("WARN", "drift", name, f"weight {w:.1%} is within 25% of its band edge"))
    return out


def risk_findings(vol: float, drift_te: float, config: PortfolioConfig) -> list[Finding]:
    out = []
    low, high = config.policy.vol_range
    if not low <= vol <= high:
        out.append(Finding("BREACH", "risk", "portfolio", f"ex-ante volatility {vol:.1%} outside {low:.0%}-{high:.0%}"))
    if drift_te > config.policy.max_drift_te:
        out.append(
            Finding(
                "WARN",
                "risk",
                "portfolio",
                f"drift tracking error {drift_te:.2%} vs model above {config.policy.max_drift_te:.2%}",
            )
        )
    return out


def concentration_findings(weights: pd.Series, config: PortfolioConfig) -> list[Finding]:
    out = []
    pol = config.policy
    for name, w in weights.items():
        if name != CASH and w > pol.max_single_weight:
            out.append(
                Finding("BREACH", "concentration", name, f"{w:.1%} in one fund exceeds {pol.max_single_weight:.0%}")
            )
    foreign = sum(
        w
        for name, w in weights.items()
        if name != CASH
        and config.instruments[name].currency != config.base_currency
        and not config.instruments[name].hedged
    )
    if foreign > pol.max_foreign_currency:
        out.append(
            Finding(
                "BREACH",
                "currency",
                "portfolio",
                f"unhedged foreign-currency exposure {foreign:.1%} exceeds {pol.max_foreign_currency:.0%}",
            )
        )
    return out


def diligence_findings(reviews: list[FundReview], switches: list[SwitchCase]) -> list[Finding]:
    out = []
    for r in reviews:
        for flag in r.flags:
            out.append(Finding("WARN", "fund review", r.ticker, flag))
        if not any("drag beyond fee" in f for f in r.flags) and r.leakage < -0.001:
            out.append(
                Finding(
                    "INFO",
                    "fund review",
                    r.ticker,
                    f"drag beyond fee of {-r.leakage:.2%} a year is not yet significant (t = {r.leakage_t:.1f})",
                )
            )
    for s in switches:
        sev = "WARN" if s.recommend else "INFO"
        out.append(Finding(sev, "replacement", s.current, f"vs {s.candidate}: {s.reason}"))
    return out


def timing_findings(timings: list[OrderTiming], calendar: Calendar, base_timezone: str) -> list[Finding]:
    """One finding per distinct hold reason or note, naming every order it touches."""
    tz = ZoneInfo(base_timezone)
    out = []
    if calendar.unavailable:
        out.append(Finding("WARN", "timing", "calendar", f"{calendar.unavailable}; every non-urgent order is held"))
    held, overrides, notes = defaultdict(list), [], defaultdict(list)
    for t in timings:
        if t.verdict == "HOLD" and not calendar.unavailable:
            release = f"{t.release_at.astimezone(tz):%d %b %H:%M} ({base_timezone})"
            held[(t.reason, release)].append(t.trade.ticker)
        elif t.reason.startswith("urgent override"):
            overrides.append(t)
        for note in t.notes:
            notes[note].append(t.trade.ticker)
    for (reason, release), tickers in held.items():
        out.append(Finding("WARN", "timing", ", ".join(tickers), f"held for {reason}; send after {release}"))
    for t in overrides:
        out.append(Finding("INFO", "timing", t.trade.ticker, t.reason))
    for note, tickers in notes.items():
        out.append(Finding("INFO", "timing", ", ".join(tickers), note))
    return out


def ranked(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.check, f.subject))
