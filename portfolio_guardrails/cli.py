"""Command line entry point.

    python -m portfolio_guardrails demo [--out reports] [--seed 3]
    python -m portfolio_guardrails run --config C --prices P --fx F --factors X \\
        --holdings H --cash 1000 --as-of 2026-09-29 --events E [--trade-date D] [--out reports]

Exit code 2 means the run was blocked by bad data, so a scheduler can alert
on it; breaches are normal business and exit 0 with trades proposed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_config
from .market import MarketData
from .monitor import run_monitoring
from .rebalance import whole_lots
from .report import write_outputs
from .synthetic import generate_market
from .timing import load_calendar, next_weekday

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_CONFIG = EXAMPLES / "balanced_sgd.toml"
EXAMPLE_CALENDAR = EXAMPLES / "calendar_2026h2.toml"


def demo_account(config, md: MarketData, as_of: pd.Timestamp) -> tuple[pd.Series, float]:
    """An account bought at target weights a year ago, left to drift, then topped up today."""
    start = md.prices.index[md.prices.index.get_loc(as_of) - 252]
    nav = config.scenario.start_nav
    units, spent = {}, 0.0
    for t in config.risky_targets:
        inst = config.instruments[t]
        fx = 1.0 if inst.currency == config.base_currency else float(md.fx.loc[start, inst.currency])
        px_base = float(md.prices.loc[start, t]) * fx
        units[t] = whole_lots(config.policy.targets[t] * nav, px_base, inst.lot_size)
        spent += units[t] * px_base
    cash = nav - spent + config.scenario.deposit  # unspent rounding stays in cash
    return pd.Series(units, dtype=float), cash


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="portfolio_guardrails")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run on a seeded synthetic market")
    d.add_argument("--config", default=str(EXAMPLE_CONFIG))
    d.add_argument("--events", default=str(EXAMPLE_CALENDAR))
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--out", default="reports")
    r = sub.add_parser("run", help="run on your own CSV data")
    r.add_argument("--config", required=True)
    r.add_argument("--prices", required=True)
    r.add_argument("--fx", required=True)
    r.add_argument("--factors", required=True)
    r.add_argument("--holdings", required=True, help="CSV with columns ticker,units")
    r.add_argument("--cash", type=float, required=True)
    r.add_argument("--as-of", required=True)
    r.add_argument("--events", required=True, help="event calendar TOML; if unreadable, every order is held")
    r.add_argument("--trade-date", help="date the orders are sent (default: next weekday after --as-of)")
    r.add_argument("--out", default="reports")
    args = p.parse_args(argv)

    config = load_config(args.config)
    if args.cmd == "demo":
        md = generate_market(config, seed=args.seed)
        as_of = md.prices.index[-1]
        units, cash = demo_account(config, md, as_of)
    else:
        md = MarketData.from_csv(args.prices, args.fx, args.factors)
        as_of = pd.Timestamp(args.as_of)
        units = pd.read_csv(args.holdings).set_index("ticker")["units"].astype(float)
        cash = args.cash

    given = getattr(args, "trade_date", None)
    trade_date = date.fromisoformat(given) if given else next_weekday(as_of.date())
    calendar = load_calendar(args.events, trade_date)
    result = run_monitoring(config, md, units, cash, as_of, calendar=calendar, trade_date=trade_date)
    for path in write_outputs(result, args.out):
        print(f"wrote {path}")
    worst = result.findings[0].severity if result.findings else "none"
    print(f"{len(result.findings)} findings, most severe: {worst}")
    return 2 if result.blocked else 0


if __name__ == "__main__":
    sys.exit(main())
