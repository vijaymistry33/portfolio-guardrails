from datetime import date, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from conftest import CASH, make_instrument, make_policy, ones
from portfolio_guardrails.config import MarketSession, PortfolioConfig
from portfolio_guardrails.rebalance import plan_rebalance
from portfolio_guardrails.timing import (
    RESOLVERS,
    Calendar,
    IndexRebalance,
    MacroEvent,
    load_calendar,
    schedule_orders,
)

TRADE_DATE = date(2026, 9, 30)
INSTS = {
    "U": make_instrument("U", market="US"),
    "S": make_instrument("S", market="SG"),
}
CONFIG = PortfolioConfig(
    "t",
    "SGD",
    3,
    INSTS,
    make_policy({"U": 0.45, "S": 0.45, CASH: 0.10}, {"U": 0.05, "S": 0.05, CASH: 0.05}),
    None,
    markets={
        "US": MarketSession("America/New_York", time(10, 0)),
        "SG": MarketSession("Asia/Singapore", time(9, 30)),
    },
    base_timezone="Asia/Singapore",
)


def plan_for(units):
    units = pd.Series(units, dtype=float)
    cash = 100.0 - units.sum()
    return plan_rebalance(units, cash, ones(list(INSTS)), ones(list(INSTS)), INSTS, CONFIG.policy)


MILD = {"U": 40, "S": 40}  # cash 20%: buys in U and S, U only 1 band from target
SEVERE = {"U": 30, "S": 40}  # U is 15 points (3 bands) under target


def us_event(t="10:00", impact="HIGH", markets=("US",), source="curated", day=TRADE_DATE):
    return MacroEvent(
        "e", "Test release", impact, markets, day, time.fromisoformat(t) if t else None, "America/New_York", source
    )


def verdicts(cal, units=MILD, trade_date=TRADE_DATE):
    return {o.trade.ticker: o for o in schedule_orders(plan_for(units), CONFIG, cal, trade_date)}


@pytest.mark.parametrize(
    "event_time, expected",
    [("10:30", "HOLD"), ("10:31", "GO"), ("09:45", "HOLD"), ("09:44", "GO"), ("10:00", "HOLD")],
)
def test_window_runs_from_30_before_to_15_after_inclusive(event_time, expected):
    assert verdicts(Calendar([us_event(event_time)]))["U"].verdict == expected


def test_hold_releases_15_minutes_after_the_event_and_reads_in_base_time():
    order = verdicts(Calendar([us_event("10:00")]))["U"]
    sgt = ZoneInfo("Asia/Singapore")
    assert order.send_at.astimezone(sgt).hour == 22  # 10:00 New York (EDT) is 22:00 Singapore
    assert order.release_at.astimezone(sgt).strftime("%H:%M") == "22:15"


def test_unknown_release_time_holds_the_whole_day():
    order = verdicts(Calendar([us_event(None)]))["U"]
    assert order.verdict == "HOLD" and "not announced" in order.reason
    assert order.release_at.date() == date(2026, 10, 1)


def test_control_unknown_time_on_another_day_does_not_hold():
    assert verdicts(Calendar([us_event(None, day=date(2026, 10, 1))]))["U"].verdict == "GO"


def test_feed_rows_are_shown_but_never_obeyed():
    order = verdicts(Calendar([us_event(source="feed")]))["U"]
    assert order.verdict == "GO" and any("not applied" in n for n in order.notes)
    assert verdicts(Calendar([us_event(source="curated")]))["U"].verdict == "HOLD"  # control


def test_only_high_impact_holds():
    assert verdicts(Calendar([us_event(impact="MEDIUM")]))["U"].verdict == "GO"


def test_markets_match_by_equality_with_global_wildcard():
    assert verdicts(Calendar([us_event(markets=("USD",))]))["U"].verdict == "GO"
    assert verdicts(Calendar([us_event(markets=("GLOBAL",))]))["U"].verdict == "HOLD"
    assert verdicts(Calendar([us_event()]))["S"].verdict == "GO"  # a US release never holds Singapore


def test_severe_breach_trades_through_a_hold():
    order = verdicts(Calendar([us_event()]), units=SEVERE)["U"]
    assert order.verdict == "GO" and order.reason.startswith("urgent override")
    assert verdicts(Calendar([us_event()]), units=MILD)["U"].verdict == "HOLD"  # control


@pytest.mark.parametrize(
    "content, why",
    [
        (None, "unreadable"),
        ('covers = ["2026-01-01", "2026-06-30"]\n', "not the trade date"),
        (
            'covers = ["2026-07-01", "2026-12-31"]\n[[index_rebalances]]\n'
            'provider = "FTSE"\nrule = "THIRD_FRIDAY_OF_MONTH"\nmonths = [9]\n',
            "unknown index-rebalance rule",
        ),
        ("covers = [", "unreadable"),
    ],
)
def test_untrustworthy_calendar_holds_every_order(tmp_path, content, why):
    path = tmp_path / "cal.toml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    cal = load_calendar(path, TRADE_DATE)
    assert why in cal.unavailable
    orders = verdicts(cal)
    assert {o.verdict for o in orders.values()} == {"HOLD"}
    assert verdicts(cal, units=SEVERE)["U"].verdict == "GO"  # a severe breach still goes


def test_control_valid_calendar_file_loads(tmp_path):
    path = tmp_path / "cal.toml"
    path.write_text('covers = ["2026-07-01", "2026-12-31"]\n', encoding="utf-8")
    cal = load_calendar(path, TRADE_DATE)
    assert cal.unavailable is None and {o.verdict for o in verdicts(cal).values()} == {"GO"}


def test_rebalance_rules_resolve_by_exact_name():
    assert RESOLVERS["THIRD_FRIDAY"](2026, 9) == date(2026, 9, 18)
    assert RESOLVERS["MONDAY_AFTER_THIRD_FRIDAY"](2026, 9) == date(2026, 9, 21)
    assert RESOLVERS["LAST_WEEKDAY_OF_MONTH"](2026, 5) == date(2026, 5, 29)  # 31 May 2026 is a Sunday


def test_rule_names_never_resolve_by_substring():
    # "THIRD_FRIDAY" is inside "MONDAY_AFTER_THIRD_FRIDAY": a substring dispatch would pick the Friday.
    dates = IndexRebalance("X", "MONDAY_AFTER_THIRD_FRIDAY", (9,)).effective_dates(date(2026, 9, 20))
    assert dates == [date(2026, 9, 21)]


def test_index_rebalance_annotates_and_never_holds():
    insts = {**INSTS, "U": make_instrument("U", market="US", index_family="FTSE")}
    cfg = PortfolioConfig(CONFIG.name, "SGD", 3, insts, CONFIG.policy, None, markets=CONFIG.markets)
    cal = Calendar(rebalances=[IndexRebalance("FTSE", "THIRD_FRIDAY", (9,))])
    plan = plan_for(MILD)
    before = {o.trade.ticker: o for o in schedule_orders(plan, cfg, cal, date(2026, 9, 16))}
    assert before["U"].verdict == "GO" and before["U"].notes == (
        "FTSE index rebalance effective 2026-09-18 (in 2 days)",
    )
    assert before["S"].notes == ()  # S has no index family
    after = {o.trade.ticker: o for o in schedule_orders(plan, cfg, cal, date(2026, 9, 21))}
    assert after["U"].notes == ()
