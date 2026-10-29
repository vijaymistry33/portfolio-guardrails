"""Event-aware trade timing: should each proposed order go at its send time, or wait?

A rebalance is rarely urgent to the minute, so there is no reason to send an
order into the half hour around a market-moving release, when spreads widen
and prices jump. The rules:

1. **Only HIGH-impact, curated events hold orders.** A row from an automated
   feed is shown as a note but never obeyed: its tagging of which markets an
   event affects has not been reviewed, and an unreviewed answer must not
   move money.
2. **The window is T-30 to T+15 minutes, both edges inclusive,** checked
   against the order's planned send time in its own market's time zone.
3. **Unknown release time holds the market for the whole day.** Treating an
   unknown time as midnight would put the window where no order ever is, so
   the guard would never fire on exactly the case it exists for.
4. **An unreadable calendar, or one that does not cover the trade date, is
   not a quiet day.** Every order is held (fail closed), and the report says why.
5. **Markets match by equality** ("US" never matches "USD"); "GLOBAL" matches
   every market.
6. **Index-rebalance days only annotate.** Effective dates of FTSE, S&P, MSCI
   or month-end bond-index rebalances bring heavy passive flows. Whether to
   trade into them is a dealer's judgement, so they inform and never block.
   Rule tokens dispatch by exact match: "THIRD_FRIDAY" must never resolve
   "MONDAY_AFTER_THIRD_FRIDAY".
7. **A severe breach goes anyway.** An order repairing drift of at least
   ``urgent_breach_multiple`` times its band is sent despite a hold, and is
   labelled as an override.

Exchange holidays are not modelled; "next session" means the next weekday.
"""

from __future__ import annotations

import calendar as _cal
import tomllib
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .config import PortfolioConfig
from .rebalance import RebalancePlan, Trade


def _last_weekday_of_month(year: int, month: int) -> date:
    d = date(year, month, _cal.monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def _monday_after_third_friday(year: int, month: int) -> date:
    return _third_friday(year, month) + timedelta(days=3)


RESOLVERS = {  # exact-match dispatch only; see rule 6
    "LAST_WEEKDAY_OF_MONTH": _last_weekday_of_month,
    "THIRD_FRIDAY": _third_friday,
    "MONDAY_AFTER_THIRD_FRIDAY": _monday_after_third_friday,
}


@dataclass(frozen=True)
class MacroEvent:
    id: str
    title: str
    impact: str  # HIGH / MEDIUM / LOW
    markets: tuple[str, ...]
    local_date: date
    local_time: time | None  # None: release time not announced
    timezone: str
    source: str = "curated"  # or "feed": shown, never obeyed

    @property
    def instant(self) -> datetime | None:
        if self.local_time is None:
            return None
        return datetime.combine(self.local_date, self.local_time, ZoneInfo(self.timezone))

    def affects(self, market: str) -> bool:
        return "GLOBAL" in self.markets or market in self.markets


@dataclass(frozen=True)
class IndexRebalance:
    provider: str
    rule: str
    months: tuple[int, ...]
    window_days: int = 3

    def effective_dates(self, around: date) -> list[date]:
        out = []
        for offset in (-1, 0, 1):
            month0 = around.month - 1 + offset
            year, month = around.year + month0 // 12, month0 % 12 + 1
            if month in self.months:
                out.append(RESOLVERS[self.rule](year, month))
        return out


@dataclass(frozen=True)
class Calendar:
    events: list[MacroEvent] = field(default_factory=list)
    rebalances: list[IndexRebalance] = field(default_factory=list)
    unavailable: str | None = None  # set when the calendar must not be trusted

    @classmethod
    def missing(cls, why: str) -> Calendar:
        return cls(unavailable=why)


def load_calendar(path, trade_date: date) -> Calendar:
    """Read a calendar file. Never raises: any problem returns a calendar that
    holds every order, with the reason attached."""
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        start, end = (date.fromisoformat(d) for d in raw["covers"])
        if not start <= trade_date <= end:
            return Calendar.missing(f"calendar covers {start} to {end}, not the trade date {trade_date}")
        events = [
            MacroEvent(
                id=e["id"],
                title=e["title"],
                impact=str(e["impact"]).upper(),
                markets=tuple(str(m).upper() for m in e["markets"]),
                local_date=date.fromisoformat(e["date"]),
                local_time=time.fromisoformat(e["time"]) if e.get("time") else None,
                timezone=str(ZoneInfo(e["timezone"])),
                source=e.get("source", "curated"),
            )
            for e in raw.get("events", [])
        ]
        rebalances = []
        for r in raw.get("index_rebalances", []):
            if r["rule"] not in RESOLVERS:
                return Calendar.missing(f"unknown index-rebalance rule {r['rule']!r}")
            rebalances.append(
                IndexRebalance(r["provider"], r["rule"], tuple(r["months"]), int(r.get("window_days", 3)))
            )
        return Calendar(events, rebalances)
    except Exception as exc:  # fail closed: an unreadable calendar holds everything
        return Calendar.missing(f"calendar unreadable ({type(exc).__name__}: {exc})")


@dataclass(frozen=True)
class OrderTiming:
    trade: Trade
    market: str
    send_at: datetime
    verdict: str  # GO or HOLD
    reason: str
    release_at: datetime | None = None
    notes: tuple[str, ...] = ()


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _hold_for(event: MacroEvent, send_at: datetime, before: timedelta, after: timedelta, session_time: time):
    """(reason, release_at) when ``event`` holds an order sent at ``send_at``, else None."""
    if event.source != "curated" or event.impact != "HIGH":
        return None
    instant = event.instant
    if instant is not None:
        if -after <= instant - send_at <= before:
            return f"{event.title} at {instant:%H:%M} {event.timezone}", instant + after
        return None
    local_day = send_at.astimezone(ZoneInfo(event.timezone)).date()
    if event.local_date == local_day:
        release = datetime.combine(next_weekday(send_at.date()), session_time, send_at.tzinfo)
        return f"{event.title}, release time not announced: whole day held", release
    return None


def schedule_orders(plan: RebalancePlan, config: PortfolioConfig, cal: Calendar, trade_date: date) -> list[OrderTiming]:
    t = config.timing
    before, after = timedelta(minutes=t.lockout_before_minutes), timedelta(minutes=t.lockout_after_minutes)
    out = []
    for trade in plan.trades:
        inst = config.instruments[trade.ticker]
        session = config.markets[inst.market]
        send_at = datetime.combine(trade_date, session.execution_time, ZoneInfo(session.timezone))

        drift = abs(float(plan.weights_before.get(trade.ticker, 0.0)) - config.policy.targets.get(trade.ticker, 0.0))
        band = config.policy.bands.get(trade.ticker, 0.0)
        urgent = band > 0 and drift >= t.urgent_breach_multiple * band

        holds, notes = [], []
        if cal.unavailable:
            holds.append((f"event calendar unavailable: {cal.unavailable}", None))
        for ev in cal.events:
            if not ev.affects(inst.market):
                continue
            if ev.source == "curated":
                hit = _hold_for(ev, send_at, before, after, session.execution_time)
                if hit:
                    holds.append(hit)
            elif _hold_for(replace(ev, source="curated"), send_at, before, after, session.execution_time):
                notes.append(f"unreviewed feed lists {ev.title} in this window; not applied")
        for rb in cal.rebalances:
            if rb.provider != inst.index_family:
                continue
            for eff in rb.effective_dates(trade_date):
                days = (eff - trade_date).days
                if 0 <= days <= rb.window_days:
                    when = "today" if days == 0 else f"in {days} day{'s' * (days > 1)}"
                    notes.append(f"{rb.provider} index rebalance effective {eff} ({when})")

        if not holds:
            out.append(OrderTiming(trade, inst.market, send_at, "GO", "no event window", None, tuple(notes)))
            continue
        reason = "; ".join(h[0] for h in holds)
        releases = [h[1] for h in holds]
        release = None if any(r is None for r in releases) else max(releases)
        if urgent:
            reason = f"urgent override (drift {drift:.1%} is {drift / band:.1f}x its band) despite: {reason}"
            out.append(OrderTiming(trade, inst.market, send_at, "GO", reason, None, tuple(notes)))
        else:
            out.append(OrderTiming(trade, inst.market, send_at, "HOLD", reason, release, tuple(notes)))
    return out
