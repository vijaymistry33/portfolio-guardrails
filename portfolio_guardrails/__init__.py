"""portfolio_guardrails: daily monitoring, rebalancing and fund due diligence
for a multi-asset, multi-currency model portfolio.

The package is organised as a pipeline of small, separately testable steps:

    config     -> load and validate the portfolio policy (fails on every problem at once)
    market     -> price/FX snapshots and base-currency returns (fails closed on bad data)
    risk       -> covariance, volatility, Euler risk contributions, tracking error, VaR
    factors    -> OLS factor exposures with Newey-West t-stats and a variance decomposition
    rebalance  -> band-triggered, cash-flow-first rebalancing with lot sizes and costs
    diligence  -> ETF tracking difference / error, liquidity, and switch break-even
    checks     -> turns all of the above into severity-ranked governance findings
    narrator   -> drafts the summary; an LLM may write it, but every number is verified
    monitor    -> runs the pipeline end to end
"""

__version__ = "1.0.0"
