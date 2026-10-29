"""Render a MonitoringResult as a Markdown report plus machine-readable files."""

from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import __version__
from .config import CASH
from .monitor import MonitoringResult


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def render_markdown(res: MonitoringResult) -> str:
    cfg, ccy = res.config, res.config.base_currency
    md = [
        f"# Monitoring report: {cfg.name}",
        f"As of **{res.as_of.date()}** · base currency {ccy} · portfolio_guardrails {__version__}",
    ]

    if res.blocked:
        md += ["", "## RUN BLOCKED: data quality", "No trades are proposed until these are resolved.", ""]
        md.append(
            _table(
                ["Severity", "Check", "Subject", "Detail"],
                [[f.severity, f.check, f.subject, f.message] for f in res.findings],
            )
        )
        return "\n".join(md) + "\n"

    plan = res.plan
    md += ["", "## Summary", "", res.narrative.text, "", f"_Summary written by: {res.narrative.source}_"]
    if res.narrative.rejected_numbers:
        md.append(f"_LLM draft rejected; unverifiable numbers: {', '.join(res.narrative.rejected_numbers)}_")

    md += ["", "## Findings", ""]
    md.append(
        _table(
            ["Severity", "Check", "Subject", "Detail"],
            [[f.severity, f.check, f.subject, f.message] for f in res.findings],
        )
    )

    md += ["", "## Allocation and drift", ""]
    rows = []
    for name in [*cfg.risky_targets, CASH]:
        t, band = cfg.policy.targets[name], cfg.policy.bands[name]
        now, after = plan.weights_before.get(name, 0.0), plan.weights_after.get(name, 0.0)
        status = "OUT" if name in plan.breaches_before else "ok"
        rows.append([name, f"{t:.1%}", f"±{band:.1%}", f"{now:.1%}", status, f"{after:.1%}"])
    md.append(_table(["Holding", "Target", "Band", "Now", "Status", "After trades"], rows))

    md += ["", f"## Proposed trades ({plan.method}) for {res.trade_date}", ""]
    if plan.trades:
        tz = ZoneInfo(cfg.base_timezone)
        timing = {t.trade.ticker: t for t in res.timings}

        def when(t) -> str:
            ot = timing[t.ticker]
            if ot.verdict == "GO":
                return "GO" + (" (urgent override)" if ot.reason.startswith("urgent") else "")
            return f"HOLD to {ot.release_at.astimezone(tz):%d %b %H:%M}" if ot.release_at else "HOLD: no calendar"

        md.append(
            _table(
                [
                    "Order",
                    "Units",
                    "Price (local)",
                    f"Value ({ccy})",
                    f"Est. cost ({ccy})",
                    f"Send at ({cfg.base_timezone})",
                    "Timing",
                ],
                [
                    [
                        f"{t.side} {t.ticker}",
                        f"{t.units:,.0f}",
                        f"{t.currency} {t.price_local:,.2f}",
                        f"{t.value_base:,.0f}",
                        f"{t.cost_base:,.2f}",
                        f"{timing[t.ticker].send_at.astimezone(tz):%d %b %H:%M}",
                        when(t),
                    ]
                    for t in plan.trades
                ],
            )
        )
        md.append("\nHold reasons and calendar notes are listed under Findings (check: timing).")
        md.append(
            f"\nNAV {ccy} {plan.nav:,.0f} · turnover {plan.turnover:.1%} · estimated cost {ccy} {plan.total_cost:,.2f} "
            f"· cash after {ccy} {plan.cash_after:,.0f}"
        )
    else:
        md.append("None.")

    lo, hi = cfg.policy.vol_range
    md += [
        "",
        "## Risk",
        "",
        f"- Ex-ante volatility: **{res.vol:.2%}** now, {res.vol_after:.2%} after trades (policy {lo:.0%}-{hi:.0%})",
        f"- Tracking error of current weights vs the model: {res.drift_te:.2%}",
        f"- 1-day 95% VaR (normal): {ccy} {res.var_95:,.0f}",
        "",
    ]
    c = res.contributions.sort_values("contribution", ascending=False)
    md.append(
        _table(
            ["Holding", "Weight", "Risk contribution", "Share of risk"],
            [[n, f"{r.weight:.1%}", f"{r.contribution:.2%}", f"{r.share:.0%}"] for n, r in c.iterrows()],
        )
    )

    fit = res.portfolio_fit
    md += ["", f"## Factor exposures ({fit.n_obs} days, Newey-West t-stats)", ""]
    md.append(
        _table(
            ["Factor", "Beta", "t-stat", "Share of variance"],
            [
                [k, f"{fit.betas[k]:.2f}", f"{fit.t_stats[k]:.1f}", f"{res.decomposition[k]:.0%}"]
                for k in fit.betas.index
            ]
            + [["specific", "", "", f"{res.decomposition['specific']:.0%}"]],
        )
    )
    md.append(f"\nR² {fit.r_squared:.2f}. Betas are of today's holdings applied to past returns (a backcast).\n")
    factors = list(fit.betas.index)
    md.append(
        _table(
            ["Fund", *factors, "R²"],
            [[t, *[f"{f.betas[k]:.2f}" for k in factors], f"{f.r_squared:.2f}"] for t, f in res.fund_fits.items()],
        )
    )

    md += ["", "## Fund due diligence", ""]
    md.append(
        _table(
            ["Fund", "Benchmark", "Years", "Tracking diff.", "TER", "Leakage (t)", "Tracking error", "Flags"],
            [
                [
                    r.ticker,
                    r.benchmark,
                    f"{r.years:.1f}",
                    f"{r.tracking_difference:+.2%}",
                    f"{r.ter:.2%}",
                    f"{r.leakage:+.2%} ({r.leakage_t:+.1f})",
                    f"{r.tracking_error:.2%}",
                    "; ".join(r.flags) or "none",
                ]
                for r in res.reviews
            ],
        )
    )
    if res.switches:
        md += ["", "### Replacement candidates", ""]
        md.append(
            _table(
                ["Current", "Candidate", "Cost now / yr", "Candidate / yr", "Switch cost", "Break-even", "Verdict"],
                [
                    [
                        s.current,
                        s.candidate,
                        f"{s.current_cost:.2%}",
                        f"{s.candidate_cost:.2%}",
                        f"{s.switch_cost:.2%}",
                        f"{s.breakeven_months:.1f} months" if s.breakeven_months != float("inf") else "never",
                        ("REVIEW: " if s.recommend else "keep: ") + s.reason,
                    ]
                    for s in res.switches
                ],
            )
        )

    md += [
        "",
        "## Method and assumptions",
        "",
        "- Covariance: EWMA (lambda 0.94) on 252 days of base-currency returns; zero daily mean.",
        "- Hedged holdings carry no FX risk; hedge carry is ignored.",
        "- Staleness is counted in weekdays; exchange holiday calendars are not modelled.",
        "- Fund leakage counts only when material (>= 5 bps a year) and |t| >= 2 (standard error TE / sqrt(years)).",
        "- Costs: commission plus half the quoted spread per order; market impact and taxes ignored.",
    ]
    return "\n".join(md) + "\n"


def write_outputs(res: MonitoringResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "report.md", out / "findings.json"]
    paths[0].write_text(render_markdown(res), encoding="utf-8")
    paths[1].write_text(json.dumps([f.to_dict() for f in res.findings], indent=2), encoding="utf-8")
    if res.plan is not None:
        trades = pd.DataFrame([t.__dict__ for t in res.plan.trades])
        paths.append(out / "trades.csv")
        trades.to_csv(paths[-1], index=False)
    return paths
