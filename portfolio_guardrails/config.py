"""Portfolio policy: instruments, targets, tolerance bands and limits.

Everything a monitoring run depends on lives in one TOML file so that a policy
change is a reviewed diff, not a code change. Validation collects *every*
problem before failing, so a bad file is fixed in one pass rather than one
error at a time.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

CASH = "CASH"


class ConfigError(ValueError):
    """Raised when the policy file is incomplete or inconsistent."""


@dataclass(frozen=True)
class Instrument:
    """One tradable vehicle (an ETF here) and the facts needed to trade and review it."""

    ticker: str
    name: str
    asset_class: str
    market: str  # listing venue, e.g. US / SG / HK / AU
    currency: str  # trading currency
    lot_size: int  # board lot: orders must be whole multiples of this
    min_trade_local: float  # smallest order value accepted, in trading currency
    commission_bps: float  # one-way commission
    hedged: bool  # FX risk hedged to base currency (hedge carry is ignored)
    benchmark: str | None  # index the fund tracks; None for no review
    ter: float  # total expense ratio, as a fraction (0.002 = 0.20 %)
    aum_usd_m: float
    adv_usd_m: float  # average daily traded value
    spread_bps: float  # typical quoted bid-ask spread
    index_family: str | None = None  # index provider, for rebalance-day notes (e.g. FTSE)

    @property
    def one_way_cost_bps(self) -> float:
        """Commission plus crossing half the spread."""
        return self.commission_bps + self.spread_bps / 2


@dataclass(frozen=True)
class Policy:
    targets: dict[str, float]  # strategic weights, including CASH; sums to 1
    bands: dict[str, float]  # absolute tolerance around each target
    min_trade_base: float  # orders smaller than this are not worth their cost
    prefer_cashflows: bool  # try to fix drift with cash before selling anything
    vol_range: tuple[float, float]  # acceptable ex-ante annualised volatility
    max_drift_te: float  # ex-ante tracking error of current vs target weights
    max_single_weight: float
    max_foreign_currency: float  # unhedged non-base currency exposure


@dataclass(frozen=True)
class DiligenceLimits:
    window_days: int
    max_tracking_error: float
    min_aum_usd_m: float
    min_adv_usd_m: float
    max_spread_bps: float
    expected_turnover: float  # traded value per year / position value
    max_breakeven_months: float


@dataclass(frozen=True)
class Scenario:
    """Demo-only: how the illustrative account came to be where it is."""

    start_nav: float
    deposit: float


@dataclass(frozen=True)
class MarketSession:
    """When a market's orders are sent: a local time in the market's own time zone."""

    timezone: str
    execution_time: time


@dataclass(frozen=True)
class TimingPolicy:
    lockout_before_minutes: int = 30
    lockout_after_minutes: int = 15
    urgent_breach_multiple: float = 2.0  # drift this many bands out trades through a hold


@dataclass(frozen=True)
class PortfolioConfig:
    name: str
    base_currency: str
    stale_after_days: int
    instruments: dict[str, Instrument]
    policy: Policy
    diligence: DiligenceLimits
    alternatives: dict[str, list[str]] = field(default_factory=dict)
    scenario: Scenario | None = None
    markets: dict[str, MarketSession] = field(default_factory=dict)
    timing: TimingPolicy = field(default_factory=TimingPolicy)
    base_timezone: str = "UTC"

    @property
    def risky_targets(self) -> list[str]:
        return [t for t in self.policy.targets if t != CASH]


def load_config(path: str | Path) -> PortfolioConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return parse_config(raw)


def parse_config(raw: dict) -> PortfolioConfig:
    problems: list[str] = []

    def need(table: dict, key: str, where: str):
        if key not in table:
            problems.append(f"{where}: missing '{key}'")
            return None
        return table[key]

    instruments: dict[str, Instrument] = {}
    for ticker, row in raw.get("instruments", {}).items():
        try:
            inst = Instrument(ticker=ticker, **row)
        except TypeError as exc:  # missing or unknown keys
            problems.append(f"instruments.{ticker}: {exc}")
            continue
        if inst.lot_size < 1:
            problems.append(f"instruments.{ticker}: lot_size must be >= 1")
        if not 0 <= inst.ter < 0.05:
            problems.append(f"instruments.{ticker}: ter {inst.ter} looks like a percent, expected a fraction")
        instruments[ticker] = inst

    pol = raw.get("policy", {})
    targets: dict[str, float] = {}
    bands: dict[str, float] = {}
    for ticker, row in pol.get("targets", {}).items():
        if "weight" not in row or "band" not in row:
            problems.append(f"policy.targets.{ticker}: needs both 'weight' and 'band'")
            continue
        targets[ticker] = float(row["weight"])
        bands[ticker] = float(row["band"])
        if ticker != CASH and ticker not in instruments:
            problems.append(f"policy.targets.{ticker}: no such instrument")
        if bands[ticker] <= 0:
            problems.append(f"policy.targets.{ticker}: band must be positive")
    if CASH not in targets:
        problems.append("policy.targets: CASH target is required")
    if targets and not math.isclose(sum(targets.values()), 1.0, abs_tol=1e-9):
        problems.append(f"policy.targets: weights sum to {sum(targets.values()):.6f}, not 1")

    vol_range = need(pol, "vol_range", "policy") or (0.0, 0.0)
    policy = Policy(
        targets=targets,
        bands=bands,
        min_trade_base=float(need(pol, "min_trade_base", "policy") or 0),
        prefer_cashflows=bool(pol.get("prefer_cashflows", True)),
        vol_range=(float(vol_range[0]), float(vol_range[1])),
        max_drift_te=float(need(pol, "max_drift_te", "policy") or 0),
        max_single_weight=float(need(pol, "max_single_weight", "policy") or 0),
        max_foreign_currency=float(need(pol, "max_foreign_currency", "policy") or 0),
    )

    dd = raw.get("diligence", {})
    try:
        diligence = DiligenceLimits(**dd)
    except TypeError as exc:
        problems.append(f"diligence: {exc}")
        diligence = None

    alternatives = {k: list(v) for k, v in raw.get("alternatives", {}).items()}
    for current, candidates in alternatives.items():
        for t in [current, *candidates]:
            if t not in instruments:
                problems.append(f"alternatives.{current}: unknown instrument {t}")
        for c in candidates:
            if (
                c in instruments
                and current in instruments
                and instruments[c].benchmark != instruments[current].benchmark
            ):
                problems.append(f"alternatives.{current}: {c} tracks a different benchmark")

    markets: dict[str, MarketSession] = {}
    for code, row in raw.get("markets", {}).items():
        try:
            markets[code] = MarketSession(str(ZoneInfo(row["timezone"])), time.fromisoformat(row["execution_time"]))
        except Exception as exc:  # bad zone name, bad time, missing key
            problems.append(f"markets.{code}: {type(exc).__name__}: {exc}")
    for inst in instruments.values():
        if inst.market not in raw.get("markets", {}):  # a row that failed to parse is reported above
            problems.append(f"instruments.{inst.ticker}: market {inst.market} has no [markets.{inst.market}] session")
    try:
        timing = TimingPolicy(**raw.get("timing", {}))
    except TypeError as exc:
        problems.append(f"timing: {exc}")
        timing = TimingPolicy()

    scenario = Scenario(**raw["scenario"]) if "scenario" in raw else None
    base_currency = need(raw, "base_currency", "config")
    base_timezone = raw.get("base_timezone", "UTC")
    try:
        ZoneInfo(base_timezone)
    except Exception:
        problems.append(f"base_timezone: unknown time zone {base_timezone!r}")

    if problems:
        raise ConfigError("invalid portfolio config:\n  - " + "\n  - ".join(problems))
    return PortfolioConfig(
        name=raw.get("name", "Unnamed portfolio"),
        base_currency=base_currency,
        stale_after_days=int(raw.get("stale_after_days", 3)),
        instruments=instruments,
        policy=policy,
        diligence=diligence,
        alternatives=alternatives,
        scenario=scenario,
        markets=markets,
        timing=timing,
        base_timezone=base_timezone,
    )
