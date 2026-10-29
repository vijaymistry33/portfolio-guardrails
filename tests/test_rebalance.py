import numpy as np
import pandas as pd
import pytest

from conftest import CASH, make_instrument, make_policy, ones
from portfolio_guardrails.checks import drift_findings
from portfolio_guardrails.config import PortfolioConfig
from portfolio_guardrails.rebalance import current_weights, out_of_band, plan_rebalance

TICKERS = ["A", "B"]
INSTS = {t: make_instrument(t) for t in TICKERS}
POLICY = make_policy({"A": 0.50, "B": 0.45, CASH: 0.05}, {"A": 0.10, "B": 0.03, CASH: 0.05})


def plan(units, cash, policy=POLICY, insts=INSTS, prices=None):
    prices = ones(list(units.index)) if prices is None else prices
    return plan_rebalance(units, cash, prices, ones(list(prices.index)), insts, policy)


def test_no_trades_while_inside_bands():
    p = plan(pd.Series({"A": 52.0, "B": 44.0}), cash=4.0)
    assert not p.triggered and p.trades == []


def test_control_same_book_trades_when_band_is_tight():
    tight = make_policy(POLICY.targets, {"A": 0.01, "B": 0.03, CASH: 0.05})
    p = plan(pd.Series({"A": 52.0, "B": 44.0}), cash=4.0, policy=tight)
    assert p.triggered and p.trades


def test_deposit_is_deployed_without_selling():
    # A rallied but sits inside its wide band; a deposit pushes cash out of its band.
    p = plan(pd.Series({"A": 55.0, "B": 40.0}), cash=25.0)
    assert p.method == "cash-flow first"
    assert all(t.side == "BUY" for t in p.trades)
    assert p.breaches_after == []


def test_falls_back_to_full_rebalance_when_cash_cannot_fix_the_breach():
    # B is below its band, there is no spare cash, and A is overweight but inside its band:
    # cash-flow-first cannot fund B, so the full plan must sell some A.
    p = plan(pd.Series({"A": 59.0, "B": 41.0}), cash=0.0)
    assert p.method == "full"
    assert any(t.side == "SELL" and t.ticker == "A" for t in p.trades)
    assert p.breaches_after == []


def test_control_prefer_cashflows_off_always_uses_full_plan():
    off = make_policy(POLICY.targets, POLICY.bands, prefer_cashflows=False)
    p = plan(pd.Series({"A": 55.0, "B": 40.0}), cash=25.0, policy=off)
    assert p.method == "full"


def test_exiting_a_fund_sells_all_of_it():
    insts = {**INSTS, "OLD": make_instrument("OLD")}
    p = plan(pd.Series({"A": 50.0, "B": 45.0, "OLD": 7.0}), cash=5.0, insts=insts)
    assert "OLD" in p.breaches_before
    assert "OLD" not in p.units_after.index


def test_orders_below_minimum_size_are_dropped():
    big_min = make_policy(POLICY.targets, {"A": 0.10, "B": 0.03, CASH: 0.01}, min_trade=50.0)
    p = plan(pd.Series({"A": 50.0, "B": 43.0}), cash=7.0, policy=big_min)
    assert p.triggered and p.trades == []  # the 2-unit fix is not worth a 50-unit minimum order


def test_board_lot_leaves_a_reported_residual_breach():
    # One HK board lot (500 units x 1.0) is 5% of this small account: B cannot get inside a 1% band.
    insts = {"A": make_instrument("A"), "B": make_instrument("B", lot=500)}
    policy = make_policy({"A": 0.50, "B": 0.45, CASH: 0.05}, {"A": 0.10, "B": 0.01, CASH: 0.10})
    p = plan(pd.Series({"A": 5000.0, "B": 3200.0}), cash=1800.0, policy=policy, insts=insts)
    assert all(t.units % 500 == 0 for t in p.trades if t.ticker == "B")
    assert "B" in p.breaches_after

    cfg = PortfolioConfig("t", "SGD", 3, insts, policy, None)
    warn = [f for f in drift_findings(p, cfg) if f.check == "residual drift"]
    assert warn and "board lot is 500" in warn[0].message


def test_accounting_identity_and_invariants_hold_on_random_books():
    rng = np.random.default_rng(0)
    insts = {
        "A": make_instrument("A", lot=1, commission=5, spread=4),
        "B": make_instrument("B", lot=100, commission=8, spread=10),
        "C": make_instrument("C", lot=500, commission=8, spread=6),
    }
    policy = make_policy(
        {"A": 0.4, "B": 0.3, "C": 0.25, CASH: 0.05}, {"A": 0.03, "B": 0.03, "C": 0.02, CASH: 0.02}, min_trade=100
    )
    for _ in range(300):
        units = pd.Series({t: float(rng.integers(0, 40) * insts[t].lot_size) for t in insts})
        prices = pd.Series({t: float(rng.uniform(0.5, 200)) for t in insts})
        cash = float(rng.uniform(0, 200_000))
        if (units * prices).sum() + cash <= 0:
            continue
        p = plan(units, cash, policy=policy, insts=insts, prices=prices)

        assert p.cash_after >= -1e-6
        assert (p.units_after >= 0).all()
        flows = sum((t.value_base if t.side == "SELL" else -t.value_base) - t.cost_base for t in p.trades)
        assert p.cash_after == pytest.approx(cash + flows, abs=1e-6)
        nav = float((units * prices).sum()) + cash
        for t in p.trades:
            assert t.units % insts[t.ticker].lot_size == 0
            if t.side == "SELL":
                assert t.units <= units[t.ticker]
                # rounding down means a sell never overshoots its target
                assert (units[t.ticker] - t.units) * prices[t.ticker] >= policy.targets[t.ticker] * nav - 1e-6


def test_weights_include_cash_and_sum_to_one():
    w, nav = current_weights(pd.Series({"A": 10.0}), 5.0, pd.Series({"A": 2.0}))
    assert nav == 25.0 and w.sum() == pytest.approx(1.0) and w[CASH] == pytest.approx(0.2)


def test_holding_without_target_is_a_breach():
    assert out_of_band(pd.Series({"A": 0.5, "B": 0.45, CASH: 0.04, "X": 0.01}), POLICY) == ["X"]
