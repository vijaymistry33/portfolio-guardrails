"""Turn the day's computed facts into a short written summary.

The rule for using an LLM here: **it narrates, it never computes.** Every
figure is calculated upstream in Python; the model is only asked to phrase
them. Before an LLM draft is accepted, every number in it is checked against
the facts it was given. One unrecognised number and the draft is discarded
for the deterministic template, and the rejection is recorded.

The check is strict on purpose. A false rejection costs a plainer paragraph;
a false acceptance puts a wrong number in front of a client or a committee.

``LLMNarrator`` takes any ``complete(prompt) -> str`` callable, so it is not
tied to one vendor and the tests can drive it with a fake model offline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Narrative:
    text: str
    source: str  # "template" or "llm"
    rejected_numbers: list[str] = field(default_factory=list)


def _flatten(obj) -> Iterable:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _flatten(v)
    else:
        yield obj


def allowed_numbers(facts: dict) -> set[str]:
    """Every way a fact may legitimately be written: as-is, as a percent,
    in thousands or millions, rounded to 0-2 decimals, with or without
    thousands separators and sign."""
    allowed: set[str] = set()
    for value in _flatten(facts):
        if isinstance(value, bool):
            continue
        if isinstance(value, str):  # dates and labels: allow each digit run ("2026", "09" and "9")
            for run in re.findall(r"\d+(?:\.\d+)?", value):
                allowed.update({run, run.lstrip("0") or "0"})
            continue
        if not isinstance(value, (int, float)):
            continue
        for scaled in (value, value * 100, value / 1e3, value / 1e6):
            for dp in (0, 1, 2):
                allowed.add(f"{abs(scaled):.{dp}f}")
    return allowed


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in ``text`` that cannot be traced to ``facts``."""
    allowed = allowed_numbers(facts)
    found = [m.lstrip("+-").replace(",", "") for m in _NUMBER.findall(text)]
    return [n for n in found if n not in allowed]


class TemplateNarrator:
    """Deterministic fallback: always correct, never eloquent."""

    def narrate(self, facts: dict) -> Narrative:
        lines = [f"{facts['portfolio']} as of {facts['as_of']}: NAV {facts['currency']} {facts['nav']:,.0f}."]
        if facts["breaches"]:
            lines.append(
                f"{facts['n_breaches']} holding(s) outside tolerance ({', '.join(facts['breaches'])}); "
                f"a {facts['rebalance_method']} rebalance of {facts['n_trades']} order(s) is proposed, "
                f"turning over {facts['turnover']:.1%} of NAV at an estimated cost of "
                f"{facts['currency']} {facts['trade_cost']:,.0f}."
            )
        else:
            lines.append("All holdings are inside their tolerance bands; no trades are proposed.")
        if facts["residual_breaches"]:
            lines.append(f"Still outside tolerance after trading: {', '.join(facts['residual_breaches'])}.")
        if facts["held_orders"]:
            why = (
                "because the event calendar is unavailable"
                if facts["calendar_unavailable"]
                else "for an event window to pass"
            )
            lines.append(
                f"{facts['n_held']} of the {facts['n_trades']} orders wait {why} ({', '.join(facts['held_orders'])})."
            )
        lines.append(
            f"Ex-ante volatility is {facts['vol']:.1%} against a {facts['vol_low']:.0%}-{facts['vol_high']:.0%} range; "
            f"the largest risk contributor is {facts['top_risk']} at {facts['top_risk_share']:.0%} of risk."
        )
        betas = ", ".join(f"{k} {v:.2f}" for k, v in facts["factor_betas"].items())
        lines.append(f"Factor betas: {betas}.")
        if facts["fund_flags"]:
            lines.append(f"Fund reviews raised flags on {', '.join(facts['fund_flags'])}.")
        for s in facts["switches"]:
            lines.append(f"Replacement review: {s}.")
        return Narrative(" ".join(lines), "template")


@dataclass
class LLMNarrator:
    complete: Callable[[str], str]
    fallback: TemplateNarrator = field(default_factory=TemplateNarrator)

    PROMPT = (
        "You write the daily monitoring note for an investment committee.\n"
        "Write at most 6 plain sentences. Use ONLY numbers that appear in FACTS, "
        "copied exactly or rounded; do not calculate, compare or infer new numbers. "
        "No advice, no forecasts.\n\nFACTS:\n{facts}"
    )

    def narrate(self, facts: dict) -> Narrative:
        try:
            draft = self.complete(self.PROMPT.format(facts=json.dumps(facts, indent=2, default=str)))
        except Exception:  # a down model must never block the report
            return self.fallback.narrate(facts)
        bad = ungrounded_numbers(draft, facts)
        if bad or not draft.strip():
            fallback = self.fallback.narrate(facts)
            return Narrative(fallback.text, "template", rejected_numbers=bad)
        return Narrative(draft.strip(), "llm")
