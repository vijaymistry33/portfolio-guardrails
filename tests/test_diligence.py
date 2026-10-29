import numpy as np
import pandas as pd
import pytest

from conftest import make_instrument
from portfolio_guardrails.config import DiligenceLimits
from portfolio_guardrails.diligence import compare_switch, review_fund

LIMITS = DiligenceLimits(
    window_days=756,
    max_tracking_error=0.01,
    min_aum_usd_m=100,
    min_adv_usd_m=1,
    max_spread_bps=15,
    expected_turnover=0.3,
    max_breakeven_months=12,
)


def fund_and_index(ter, leakage=0.0, te=0.0, n=757, seed=0):
    rng = np.random.default_rng(seed)
    idx_r = rng.normal(0.0003, 0.01, n - 1)
    fund_r = idx_r - (ter + leakage) / 252 + rng.normal(0, te / np.sqrt(252), n - 1)
    dates = pd.bdate_range("2023-01-02", periods=n)
    level = lambda r: 100 * np.cumprod(np.r_[1.0, 1 + r])  # noqa: E731
    return pd.DataFrame({"F": level(fund_r), "IDX": level(idx_r)}, index=dates)


def review(prices, **inst_kw):
    inst = make_instrument("F", benchmark="IDX", **inst_kw)
    return review_fund(inst, prices, prices.index[-1], LIMITS)


def test_fee_only_drag_gives_td_about_minus_ter_and_no_leakage():
    r = review(fund_and_index(ter=0.005), ter=0.005)
    assert r.tracking_difference == pytest.approx(-0.005, abs=0.0005)
    assert abs(r.leakage) < 0.0005 and r.tracking_error < 1e-9
    assert r.flags == []


def test_real_leakage_is_flagged_when_tracking_is_tight():
    r = review(fund_and_index(ter=0.002, leakage=0.006, te=0.002), ter=0.002)
    assert r.leakage_t <= -2
    assert any("drag beyond fee" in f for f in r.flags)


def test_control_same_leakage_is_not_flagged_when_tracking_is_noisy():
    r = review(fund_and_index(ter=0.002, leakage=0.006, te=0.03), ter=0.002)
    assert not any("drag beyond fee" in f for f in r.flags)
    assert any("tracking error" in f for f in r.flags)


def test_size_liquidity_and_spread_flags():
    prices = fund_and_index(ter=0.001)
    small = review(prices, ter=0.001, aum_usd_m=40, adv_usd_m=0.5, spread=25)
    assert len(small.flags) == 3
    assert review(prices, ter=0.001, aum_usd_m=4000, adv_usd_m=50, spread=3).flags == []


def test_switch_breakeven_arithmetic():
    prices = fund_and_index(ter=0.0068)
    prices["G"] = fund_and_index(ter=0.0015)["F"]
    cur_i = make_instrument("F", benchmark="IDX", ter=0.0068, commission=3, spread=5)
    cand_i = make_instrument("G", benchmark="IDX", ter=0.0015, commission=3, spread=3)
    insts = {"F": cur_i, "G": cand_i}
    case = compare_switch(
        review_fund(cur_i, prices, prices.index[-1], LIMITS),
        review_fund(cand_i, prices, prices.index[-1], LIMITS),
        insts,
        LIMITS,
    )
    # saving = fee gap 0.53% + 0.3 x (5.5 - 4.5) bps; switch = 5.5 + 4.5 bps
    saving = 0.0053 + 0.3 * 1e-4
    assert case.annual_saving == pytest.approx(saving)
    assert case.breakeven_months == pytest.approx(0.0010 / saving * 12)
    assert case.recommend


def test_switch_not_recommended_into_a_flagged_or_dearer_fund():
    prices = fund_and_index(ter=0.002)
    prices["G"] = prices["F"]
    insts = {
        "F": make_instrument("F", benchmark="IDX", ter=0.002),
        "G": make_instrument("G", benchmark="IDX", ter=0.002, aum_usd_m=10),
    }
    rv = {t: review_fund(insts[t], prices, prices.index[-1], LIMITS) for t in insts}
    flagged = compare_switch(rv["F"], rv["G"], insts, LIMITS)
    assert not flagged.recommend and "own flags" in flagged.reason
    insts["G"] = make_instrument("G", benchmark="IDX", ter=0.004)
    rv["G"] = review_fund(insts["G"], prices, prices.index[-1], LIMITS)
    assert not compare_switch(rv["F"], rv["G"], insts, LIMITS).recommend
