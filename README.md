# portfolio-guardrails

Daily monitoring, rebalancing and ETF due diligence for a multi-asset,
multi-currency model portfolio, written as one auditable Python pipeline.

Every day it answers the questions an investment team asks of a managed
portfolio:

| Question | Module | Method |
|---|---|---|
| Is the portfolio where the model says it should be? | `rebalance.py` | Tolerance bands, cash-flow-first rebalancing, board lots, minimum orders, costs |
| How much risk is it running, and from where? | `risk.py` | EWMA covariance, Euler risk contributions, drift tracking error, VaR |
| What actually drives its returns? | `factors.py` | OLS factor betas, Newey-West t-stats, variance decomposition |
| Is each ETF doing its job? Is a replacement worth it? | `diligence.py` | Tracking difference/error, fee leakage with a significance test, liquidity, switch break-even |
| When should each order actually be sent? | `timing.py` | Holds orders around high-impact releases per market; fails closed without a trusted calendar |
| What needs attention today? | `checks.py` | Severity-ranked findings; bad data blocks the run |
| Can AI write the daily note safely? | `narrator.py` | The LLM narrates, never computes; every number is checked against the facts |

The example is an SGD-based balanced portfolio holding funds listed in the US,
Singapore, Hong Kong and Australia. That brings in FX translation, an
FX-hedged bond sleeve, HKEX board lots, an ASX minimum order size, and four
time zones for sending orders.

## Run it

Python 3.11+, with numpy and pandas only. Tested on Python 3.13, numpy 2.2 and pandas 3.0.

```bash
pip install -e ".[dev]"
python -m portfolio_guardrails demo     # writes reports/report.md, findings.json, trades.csv
python -m pytest -q                     # 72 tests, about 2 seconds
python tools/mutation_check.py          # plants 16 bugs, confirms the tests catch every one
```

To run it on your own data, pass CSV files of prices, FX and factors (one row per date), a holdings file
and an event calendar (same format as [`examples/calendar_2026h2.toml`](examples/calendar_2026h2.toml)):

```bash
python -m portfolio_guardrails run --config examples/balanced_sgd.toml \
    --prices prices.csv --fx fx.csv --factors factors.csv \
    --holdings holdings.csv --cash 5000 --as-of 2026-09-29 --events calendar.toml
```

The exit code is `2` when bad data blocks the run, so a scheduler can raise an alert.
Policy breaches exit `0`, with trades proposed, because breaches are routine business.

## What the demo shows

The demo account was invested at its target weights a year ago with SGD 250,000
and left to drift. Today an SGD 30,000 deposit lands.
[`examples/sample_report.md`](examples/sample_report.md) is the full output. Highlights:

- **Cash at 14.3% and GEQ at 26.4% are outside their bands.** The deposit fixes both without
  a single sale: 6 buys, 8.9% turnover, about SGD 16 of estimated cost. Every holding ends inside its band.
- **Volatility is 7.9% before trading and 8.7% after,** inside the 7–12% range. GEQ carries 51% of the
  risk at 26% of the weight.
- **Factor betas: equity 0.47, rates 0.35, USD/SGD 0.45.** Nearly half the portfolio moves with the US
  dollar, which matters for an SGD investor holding unhedged US-listed equity.
- **Fund review:** AUQ is flagged as too small, too thin and too wide-spread. EMQ's 1.56% tracking error
  is flagged. The same-index EMQ_ALT would save 0.53% a year and repay the switch in 2.3 months.
- **4 of the 6 orders wait.** The three US orders would go at 10:00 New York (22:00 Singapore), exactly
  when a Fed Chair testimony starts, so they are held to 22:15. The Australian order waits a day because
  a high-impact statement there has no announced time. The Hong Kong order goes: the China PMI came out
  30 minutes before it, outside the window. An unreviewed feed row (US crude inventories at 10:30) is
  shown, not obeyed. IGB carries a note that its bond index rebalances at today's month end.

The market is synthetic, seeded and offline. Factor t-stats look very large because the
returns are built from those exact factors; on real data, expect far smaller t-stats.
The default seed (3) gives an ordinary year: +8% on the invested money.
Seed 7 gives a +49% equity rally, a 2.6-sigma year, which is not a fair showcase.

## Design decisions worth reviewing

1. **Fail closed on data.** A missing, non-positive or stale price or FX rate raises
   `DataQualityError` listing every problem at once. The run then proposes **no trades**. Carrying a
   stale price forward silently is how a rebalancer ends up trading on yesterday's market.
2. **Cash-flow first, then full.** Deposits and withdrawals are the cheapest way to rebalance.
   The planner first tries to fix drift using cash and only the breached overweights. If that leaves
   more breaches than a full rebalance would, it falls back to the full plan, which also covers
   withdrawals and drift with no cash available.
3. **Round down, never oversell, never overspend.** Orders are whole board lots, rounded toward zero.
   A randomized test checks 300 books for non-negative cash and positions, whole lots, no sell beyond
   holdings or past target, and an exact cash accounting identity.
4. **Report what cannot be fixed.** When a single HKEX board lot is a big slice of a small account,
   the residual breach shows up as a finding naming the lot size. It is never hidden.
5. **Don't flag noise.** A fund's realised tracking difference over three years has a standard error of
   about TE / √years. Leakage beyond the fee becomes a finding only when it is material (≥ 5 bps) *and*
   at least two standard errors from zero. Leakage is measured against a fee-only fund's expected
   tracking difference, −TER × (1 + index return), so fee compounding in a strong year is not mistaken
   for leakage. The demo plants 0.4% of leakage in EMQ, and the tool correctly declines to call it:
   1.6% tracking error makes three years too short to tell it from noise.
6. **Switches are judged on cost to hold plus cost to move.** That means the fee, real leakage and
   expected trading, against the one-off cost of selling and buying, expressed as a break-even in
   months. A candidate with its own flags is never recommended.
7. **The LLM narrates; it never computes.** `LLMNarrator` accepts any `complete(prompt) -> str` callable,
   so it is vendor-neutral and testable offline. Every number in a draft is checked against the facts
   the model was given, in any legitimate format (`0.0786` → `7.9%`, `301114.27` → `301,114` or `301k`).
   One unrecognised number and the draft is replaced by the deterministic template, with the rejection
   logged. The deterministic template is itself tested against the same check.
8. **Policy is data.** Targets, bands, limits, lot sizes and fees live in one TOML file. The validator
   reports every problem in one pass, and it catches a TER typed as a percent (`0.68` instead of `0.0068`).
9. **Don't send orders into a release.** These rules carry over, as fresh code, from a personal trading
   console I built, where the same guard holds trades around macro releases:
   - Orders are held from 30 minutes before to 15 minutes after a HIGH-impact release in their market,
     measured at the order's send time in that market's own time zone.
   - A release with no announced time holds that market for the whole day. Treating "unknown" as
     midnight would put the window where no order ever is, so the guard would never fire in the one
     case it exists for.
   - A missing or unreadable calendar, or one that does not cover the trade date, holds every order
     and says so. A calendar that cannot be read is not a quiet day.
   - Only curated rows are obeyed. Automated feed rows are shown but never obeyed, because nobody has
     reviewed which markets they claim to affect.
   - Markets match by equality (`US` never matches `USD`), and rule names dispatch exactly
     (`THIRD_FRIDAY` must never resolve `MONDAY_AFTER_THIRD_FRIDAY`).
   - Index-rebalance days only annotate, since trading into passive flows is a dealer's judgement.
   - An order repairing drift of at least two bands still goes. The console halves position size
     during a lockout instead, but a rebalance is rarely urgent to the minute, so deferring fits better here.

## Testing approach

- **Planted truth, not "it ran".** The factor model must recover betas planted in synthetic data.
  A scrambled-returns control must find none.
- **Identities, not snapshots.** Euler contributions must sum to portfolio volatility on random
  covariances. The factor variance shares must add back to the sample variance exactly. Cash must
  reconcile to the cent across random rebalances.
- **Every positive test has a control that can fail.** Examples: a tighter band makes the same book
  trade; the same leakage is not flagged when tracking is noisy; an unmodified config parses.
- **Mutation check.** `tools/mutation_check.py` plants 16 realistic bugs (no lot rounding, sells past
  target, staleness ignored, grounding check off, market codes matched by substring, a calendar that
  fails open, and others) and confirms the suite fails on each.
  The first version of the suite missed the "sells past target" bug. That miss is why the invariant
  in point 3 exists.

## Known simplifications

- Staleness counts weekdays; exchange holiday calendars (SGX, HKEX, ASX, NYSE) are not modelled.
  "Next session" means the next weekday.
- Each market sends its orders at one fixed local time; lunch breaks and auctions are not modelled.
- FX hedges are assumed costless: hedge carry, the interest-rate differential, is ignored.
- Costs are commission plus half the spread. Market impact and taxes are out of scope.
- VaR is parametric-normal, a gauge rather than a capital number.
- Rebalancing is rule-based. An optimiser (minimise tracking error subject to turnover and lot
  constraints) is the natural next step for larger books.

## How this was built

Developed with an AI coding assistant (Claude Code) as a pair programmer.
The verification layer exists to hold AI-written code to account rather than trust it: planted ground
truth, negative controls and mutation testing. The same principle drives `narrator.py`.

## Layout

```
portfolio_guardrails/   config, market, risk, factors, rebalance, diligence, timing, checks, narrator, monitor, report, cli, synthetic
tests/                  72 tests: unit, property, end-to-end, CLI exit codes, CSV round trip
tools/mutation_check.py planted-bug check for the test suite
examples/               policy (balanced_sgd.toml), event calendar (calendar_2026h2.toml), demo output (sample_report.md)
```
