from pathlib import Path

import pandas as pd
import pytest

from portfolio_guardrails.cli import demo_account
from portfolio_guardrails.config import CASH, Instrument, Policy, load_config
from portfolio_guardrails.synthetic import generate_market

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "balanced_sgd.toml"


@pytest.fixture(scope="session")
def config():
    return load_config(EXAMPLE)


@pytest.fixture(scope="session")
def market(config):
    return generate_market(config, seed=3)


@pytest.fixture()
def account(config, market):
    as_of = market.prices.index[-1]
    units, cash = demo_account(config, market, as_of)
    return units, cash, as_of


def make_instrument(
    ticker, lot=1, currency="SGD", min_local=0.0, commission=0.0, spread=0.0, benchmark=None, market="SG", **facts
):
    base = dict(ter=0.001, aum_usd_m=1000.0, adv_usd_m=10.0)
    base.update(facts)
    return Instrument(
        ticker=ticker,
        name=ticker,
        asset_class="Equity",
        market=market,
        currency=currency,
        lot_size=lot,
        min_trade_local=min_local,
        commission_bps=commission,
        hedged=False,
        benchmark=benchmark,
        spread_bps=spread,
        **base,
    )


def make_policy(targets, bands, min_trade=0.0, prefer_cashflows=True):
    return Policy(
        targets=targets,
        bands=bands,
        min_trade_base=min_trade,
        prefer_cashflows=prefer_cashflows,
        vol_range=(0.0, 1.0),
        max_drift_te=1.0,
        max_single_weight=1.0,
        max_foreign_currency=1.0,
    )


def ones(tickers):
    return pd.Series(1.0, index=tickers)


__all__ = ["CASH", "make_instrument", "make_policy", "ones"]
