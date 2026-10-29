import copy
import tomllib

import numpy as np
import pytest

from conftest import EXAMPLE
from portfolio_guardrails.checks import SEVERITY_ORDER
from portfolio_guardrails.cli import main
from portfolio_guardrails.config import ConfigError, parse_config
from portfolio_guardrails.market import MarketData
from portfolio_guardrails.monitor import run_monitoring
from portfolio_guardrails.report import render_markdown
from portfolio_guardrails.timing import load_calendar, next_weekday

EXAMPLE_CALENDAR = EXAMPLE.parent / "calendar_2026h2.toml"


def raw_config():
    with open(EXAMPLE, "rb") as fh:
        return tomllib.load(fh)


def test_demo_run_is_clean_and_ranked(config, market, account):
    units, cash, as_of = account
    res = run_monitoring(config, market, units, cash, as_of)
    assert not res.blocked and res.plan.triggered
    ranks = [SEVERITY_ORDER[f.severity] for f in res.findings]
    assert ranks == sorted(ranks)
    report = render_markdown(res)
    for section in ["## Findings", "## Proposed trades", "## Risk", "## Factor exposures", "## Fund due diligence"]:
        assert section in report


def test_demo_calendar_verdicts(config, market, account):
    units, cash, as_of = account
    trade_date = next_weekday(as_of.date())
    res = run_monitoring(config, market, units, cash, as_of, load_calendar(EXAMPLE_CALENDAR, trade_date), trade_date)
    got = {t.trade.ticker: t.verdict for t in res.timings}
    # US orders sit on the Fed testimony; the China PMI came 30 minutes before the HK send; the AU
    # statement has no announced time; nothing is scheduled for Singapore.
    assert got == {"GEQ": "HOLD", "IGB": "HOLD", "GLD": "HOLD", "AUQ": "HOLD", "HKQ": "GO", "SGB": "GO"}


def test_control_no_calendar_holds_every_order(config, market, account):
    units, cash, as_of = account
    res = run_monitoring(config, market, units, cash, as_of)
    assert {t.verdict for t in res.timings} == {"HOLD"}
    assert any(f.subject == "calendar" for f in res.findings)


def test_config_requires_a_session_for_every_market():
    raw = copy.deepcopy(raw_config())
    del raw["markets"]["AU"]
    with pytest.raises(ConfigError, match="market AU has no"):
        parse_config(raw)


def test_stale_price_blocks_the_run_and_proposes_nothing(config, market, account):
    units, cash, as_of = account
    prices = market.prices.copy()
    prices.loc[prices.index[-5:], "HKQ"] = np.nan  # five sessions without a print
    res = run_monitoring(config, MarketData(prices, market.fx, market.factors), units, cash, as_of)
    assert res.blocked and res.plan is None
    assert any(f.severity == "CRITICAL" and f.subject == "HKQ" for f in res.findings)
    assert "RUN BLOCKED" in render_markdown(res)


def test_missing_fx_blocks_the_run(config, market, account):
    units, cash, as_of = account
    res = run_monitoring(
        config, MarketData(market.prices, market.fx.drop(columns="AUD"), market.factors), units, cash, as_of
    )
    assert res.blocked and any("AUD" in f.message for f in res.findings)


def test_cli_exit_codes(tmp_path, monkeypatch, config, market):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "report.md").exists() and (tmp_path / "trades.csv").exists()

    import portfolio_guardrails.cli as cli

    def stale_market(cfg, seed):
        prices = market.prices.copy()
        prices.loc[prices.index[-5:], "GEQ"] = np.nan
        return MarketData(prices, market.fx, market.factors)

    monkeypatch.setattr(cli, "generate_market", stale_market)
    assert main(["demo", "--out", str(tmp_path / "blocked")]) == 2


def test_run_on_csv_files_matches_the_in_memory_run(tmp_path, config, market, account):
    units, cash, as_of = account
    market.prices.to_csv(tmp_path / "prices.csv")
    market.fx.to_csv(tmp_path / "fx.csv")
    market.factors.to_csv(tmp_path / "factors.csv")
    units.rename_axis("ticker").rename("units").to_csv(tmp_path / "holdings.csv")
    args = ["run", "--config", str(EXAMPLE), "--cash", repr(cash), "--as-of", str(as_of.date())]
    for name in ["prices", "fx", "factors", "holdings"]:
        args += [f"--{name}", str(tmp_path / f"{name}.csv")]
    args += ["--events", str(EXAMPLE_CALENDAR)]
    assert main([*args, "--out", str(tmp_path / "out")]) == 0

    trade_date = next_weekday(as_of.date())
    cal = load_calendar(EXAMPLE_CALENDAR, trade_date)
    expected = render_markdown(run_monitoring(config, market, units, cash, as_of, cal, trade_date))
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8") == expected


def test_config_reports_every_problem_at_once():
    raw = copy.deepcopy(raw_config())
    raw["policy"]["targets"]["GEQ"]["weight"] = 0.31  # weights no longer sum to 1
    raw["instruments"]["EMQ"]["ter"] = 0.68  # a percent typed as a fraction
    raw["alternatives"]["EMQ"] = ["GEQ"]  # tracks a different index
    with pytest.raises(ConfigError) as exc:
        parse_config(raw)
    msg = str(exc.value)
    assert "sum to" in msg and "looks like a percent" in msg and "different benchmark" in msg


def test_control_unmodified_config_parses():
    assert parse_config(raw_config()).name.startswith("Balanced")
