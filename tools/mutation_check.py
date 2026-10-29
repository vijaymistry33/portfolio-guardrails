"""Plant one known bug at a time in a throwaway copy and confirm the tests catch it.

A green test suite only means something if it can go red. Each mutant below is
a realistic mistake; if the suite still passes with it, a test is missing.

    python tools/mutation_check.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTANTS = [
    (
        "no board-lot rounding",
        "rebalance.py",
        "return math.floor(value_base / px_base / lot + _EPS) * lot",
        "return value_base / px_base",
    ),
    (
        "sells may overshoot target",
        "rebalance.py",
        "qty = min(whole_lots(-gap[n], px_base[n], inst.lot_size), held[n])",
        "qty = whole_lots(-gap[n], px_base[n], inst.lot_size) + inst.lot_size",
    ),
    ("cash-flow-first never chosen", "rebalance.py", "if len(lean[3]) <= len(full[3]):", "if False:"),
    ("Euler contributions not divided by vol", "risk.py", "contrib = w * cov_w / vol", "contrib = w * cov_w"),
    ("Newey-West lags ignored", "factors.py", "for lag in range(1, lags + 1):", "for lag in range(1, 1):"),
    ("staleness not enforced", "market.py", "if lag > stale_after_days:", "if False:"),
    (
        "CSV loader drops the last FX row",
        "market.py",
        "return cls(prices=read(prices), fx=read(fx), factors=read(factors))",
        "return cls(prices=read(prices), fx=read(fx).iloc[:-1], factors=read(factors))",
    ),
    (
        "leakage significance test removed",
        "diligence.py",
        "return abs(leakage) >= MATERIAL_LEAKAGE and abs(t) >= 2",
        "return abs(leakage) >= MATERIAL_LEAKAGE",
    ),
    ("LLM grounding check disabled", "narrator.py", "return [n for n in found if n not in allowed]", "return []"),
    (
        "markets matched by substring",
        "timing.py",
        'return "GLOBAL" in self.markets or market in self.markets',
        'return "GLOBAL" in self.markets or any(market in m for m in self.markets)',
    ),
    (
        "event window edges exclusive",
        "timing.py",
        "if -after <= instant - send_at <= before:",
        "if -after < instant - send_at < before:",
    ),
    ("unknown release time ignored", "timing.py", "if event.local_date == local_day:", "if False:"),
    ("unreviewed feed rows obeyed", "timing.py", 'if ev.source == "curated":', "if True:"),
    ("unreadable calendar fails open", "timing.py", "if cal.unavailable:", "if False:"),
    (
        "rebalance rules dispatched by substring",
        "timing.py",
        "out.append(RESOLVERS[self.rule](year, month))",
        "out.append(next(f for k, f in RESOLVERS.items() if k in self.rule)(year, month))",
    ),
    ("severe-breach override removed", "timing.py", "        if urgent:", "        if False:"),
]


def main() -> int:
    missed = 0
    for label, fname, old, new in MUTANTS:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("reports", "__pycache__", ".*_cache", ".git"))
            target = work / "portfolio_guardrails" / fname
            text = target.read_text(encoding="utf-8")
            if text.count(old) != 1:
                print(f"[SETUP ERROR] {label}: anchor not found exactly once in {fname}")
                missed += 1
                continue
            target.write_text(text.replace(old, new), encoding="utf-8", newline="")
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=work,
                env={**os.environ, "PYTHONPATH": str(work)},  # test the mutant, not an installed copy
                capture_output=True,
                text=True,
            )
            caught = run.returncode != 0
            missed += not caught
            print(f"[{'CAUGHT' if caught else 'MISSED'}] {label}: {run.stdout.strip().splitlines()[-1]}")
    print(f"{len(MUTANTS) - missed}/{len(MUTANTS)} planted bugs caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
