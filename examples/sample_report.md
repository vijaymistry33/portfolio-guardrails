# Monitoring report: Balanced (illustrative)
As of **2026-09-29** · base currency SGD · portfolio_guardrails 1.0.0

## Summary

Balanced (illustrative) as of 2026-09-29: NAV SGD 301,114. 2 holding(s) outside tolerance (CASH, GEQ); a cash-flow first rebalance of 6 order(s) is proposed, turning over 8.9% of NAV at an estimated cost of SGD 16. 4 of the 6 orders wait for an event window to pass (GEQ, IGB, GLD, AUQ). Ex-ante volatility is 7.9% against a 7%-12% range; the largest risk contributor is GEQ at 51% of risk. Factor betas: MKT 0.47, RATES 0.35, USD 0.45. Fund reviews raised flags on AUQ, EMQ. Replacement review: EMQ -> EMQ_ALT: saves 0.53% a year; switch repaid in 2.3 months.

_Summary written by: template_

## Findings

| Severity | Check | Subject | Detail |
|---|---|---|---|
| BREACH | drift | CASH | weight 14.3% vs target 5.0% (band +/-3.0%) |
| BREACH | drift | GEQ | weight 26.4% vs target 30.0% (band +/-3.0%) |
| WARN | drift | IGB | weight 27.7% is within 25% of its band edge |
| WARN | fund review | AUQ | AUM US$45m below US$100m (closure risk) |
| WARN | fund review | AUQ | daily traded value US$0.8m below US$1.0m |
| WARN | fund review | AUQ | spread 18 bps above 15 bps |
| WARN | fund review | EMQ | tracking error 1.56% above 1.00% |
| WARN | replacement | EMQ | vs EMQ_ALT: saves 0.53% a year; switch repaid in 2.3 months |
| WARN | timing | AUQ | held for Australian fiscal statement, release time not announced: whole day held; send after 01 Oct 08:30 (Asia/Singapore) |
| WARN | timing | GEQ, IGB, GLD | held for US Fed Chair testimony at 10:00 America/New_York; send after 30 Sep 22:15 (Asia/Singapore) |
| INFO | fund review | AUQ | drag beyond fee of 0.20% a year is not yet significant (t = -1.2) |
| INFO | fund review | GLD | drag beyond fee of 0.11% a year is not yet significant (t = -1.0) |
| INFO | timing | GEQ, IGB, GLD | unreviewed feed lists US crude inventories in this window; not applied |
| INFO | timing | IGB | Bloomberg index rebalance effective 2026-09-30 (today) |

## Allocation and drift

| Holding | Target | Band | Now | Status | After trades |
|---|---|---|---|---|---|
| GEQ | 30.0% | ±3.0% | 26.4% | OUT | 30.0% |
| EMQ | 8.0% | ±2.0% | 8.2% | ok | 8.2% |
| HKQ | 7.0% | ±2.0% | 5.9% | ok | 6.6% |
| AUQ | 5.0% | ±1.5% | 4.4% | ok | 5.0% |
| IGB | 30.0% | ±3.0% | 27.7% | ok | 29.9% |
| SGB | 10.0% | ±2.0% | 9.2% | ok | 10.0% |
| GLD | 5.0% | ±1.5% | 4.1% | ok | 4.9% |
| CASH | 5.0% | ±3.0% | 14.3% | OUT | 5.4% |

## Proposed trades (cash-flow first) for 2026-09-30

| Order | Units | Price (local) | Value (SGD) | Est. cost (SGD) | Send at (Asia/Singapore) | Timing |
|---|---|---|---|---|---|---|
| BUY GEQ | 63 | USD 122.04 | 10,850 | 4.34 | 30 Sep 22:00 | HOLD to 30 Sep 22:15 |
| BUY IGB | 96 | USD 50.70 | 6,869 | 2.40 | 30 Sep 22:00 | HOLD to 30 Sep 22:15 |
| BUY HKQ | 500 | HKD 24.46 | 2,213 | 2.21 | 30 Sep 10:00 | GO |
| BUY GLD | 8 | USD 209.70 | 2,367 | 0.83 | 30 Sep 22:00 | HOLD to 30 Sep 22:15 |
| BUY SGB | 2,200 | SGD 1.12 | 2,466 | 2.96 | 30 Sep 09:30 | GO |
| BUY AUQ | 57 | AUD 37.87 | 1,891 | 3.21 | 30 Sep 08:30 | HOLD to 01 Oct 08:30 |

Hold reasons and calendar notes are listed under Findings (check: timing).

NAV SGD 301,114 · turnover 8.9% · estimated cost SGD 15.96 · cash after SGD 16,357

## Risk

- Ex-ante volatility: **7.86%** now, 8.68% after trades (policy 7%-12%)
- Tracking error of current weights vs the model: 0.88%
- 1-day 95% VaR (normal): SGD 2,451

| Holding | Weight | Risk contribution | Share of risk |
|---|---|---|---|
| GEQ | 26.4% | 4.03% | 51% |
| EMQ | 8.2% | 1.53% | 20% |
| HKQ | 5.9% | 0.91% | 12% |
| AUQ | 4.4% | 0.73% | 9% |
| IGB | 27.7% | 0.63% | 8% |
| SGB | 9.2% | 0.01% | 0% |
| GLD | 4.1% | 0.00% | 0% |

## Factor exposures (756 days, Newey-West t-stats)

| Factor | Beta | t-stat | Share of variance |
|---|---|---|---|
| MKT | 0.47 | 147.9 | 90% |
| RATES | 0.35 | 37.7 | 1% |
| USD | 0.45 | 40.8 | 5% |
| specific |  |  | 4% |

R² 0.96. Betas are of today's holdings applied to past returns (a backcast).

| Fund | MKT | RATES | USD | R² |
|---|---|---|---|---|
| AUQ | 1.05 | 0.04 | 0.70 | 0.70 |
| EMQ | 1.08 | 0.02 | 0.95 | 0.76 |
| GEQ | 1.00 | 0.00 | 1.01 | 0.98 |
| GLD | 0.01 | 0.24 | 0.45 | 0.04 |
| HKQ | 1.01 | -0.11 | 0.93 | 0.56 |
| IGB | 0.05 | 1.00 | 0.01 | 0.97 |
| SGB | -0.00 | 0.69 | -0.02 | 0.80 |

## Fund due diligence

| Fund | Benchmark | Years | Tracking diff. | TER | Leakage (t) | Tracking error | Flags |
|---|---|---|---|---|---|---|---|
| AUQ | IDX_AU | 3.0 | -0.30% | 0.10% | -0.20% (-1.2) | 0.30% | AUM US$45m below US$100m (closure risk); daily traded value US$0.8m below US$1.0m; spread 18 bps above 15 bps |
| EMQ | IDX_EM | 3.0 | -0.45% | 0.68% | +0.26% (+0.3) | 1.56% | tracking error 1.56% above 1.00% |
| GEQ | IDX_DM | 3.0 | -0.23% | 0.20% | -0.02% (-0.2) | 0.24% | none |
| GLD | IDX_GOLD | 3.0 | -0.53% | 0.40% | -0.11% (-1.0) | 0.21% | none |
| HKQ | IDX_HK | 3.0 | -0.12% | 0.10% | -0.01% (-0.1) | 0.28% | none |
| IGB | IDX_AGG | 3.0 | +0.09% | 0.05% | +0.14% (+0.8) | 0.31% | none |
| SGB | IDX_SG | 3.0 | +0.03% | 0.24% | +0.27% (+1.2) | 0.39% | none |

### Replacement candidates

| Current | Candidate | Cost now / yr | Candidate / yr | Switch cost | Break-even | Verdict |
|---|---|---|---|---|---|---|
| EMQ | EMQ_ALT | 0.70% | 0.16% | 0.10% | 2.3 months | REVIEW: saves 0.53% a year; switch repaid in 2.3 months |

## Method and assumptions

- Covariance: EWMA (lambda 0.94) on 252 days of base-currency returns; zero daily mean.
- Hedged holdings carry no FX risk; hedge carry is ignored.
- Staleness is counted in weekdays; exchange holiday calendars are not modelled.
- Fund leakage counts only when material (>= 5 bps a year) and |t| >= 2 (standard error TE / sqrt(years)).
- Costs: commission plus half the quoted spread per order; market impact and taxes ignored.
