"""Ex-ante risk: covariance, volatility, Euler risk contributions, tracking error, VaR.

All volatilities are annualised from daily returns with 252 trading days.
Weights passed in here are *risky-asset* weights expressed as fractions of
total NAV; cash is excluded because it carries no market risk in base currency.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics-style EWMA covariance, annualised.

    Daily means are taken as zero (standard for daily data: the mean is tiny
    against the noise and estimating it adds error). Weights decay by ``lam``
    per day and are normalised to sum to one.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must be in (0, 1)")
    x = returns.to_numpy(dtype=float)
    if len(x) < 2 or np.isnan(x).any():
        raise ValueError("need at least two complete rows of returns")
    w = lam ** np.arange(len(x) - 1, -1, -1)  # newest row gets weight 1
    w /= w.sum()
    cov = (x * w[:, None]).T @ x * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _aligned(weights: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    unknown = set(weights.index) - set(cov.index)
    if unknown:  # fail closed: an asset with no risk estimate is not zero risk
        raise KeyError(f"no covariance for {sorted(unknown)}")
    return weights.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = _aligned(weights, cov)
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility.

    contribution_i = w_i * (Cov w)_i / vol, and the contributions sum exactly
    to vol. A negative contribution means the asset is diversifying.
    """
    w = _aligned(weights, cov)
    cov_w = cov.to_numpy() @ w
    vol = float(np.sqrt(w @ cov_w))
    if vol == 0:
        contrib = np.zeros_like(w)
    else:
        contrib = w * cov_w / vol
    out = pd.DataFrame(
        {"weight": w, "contribution": contrib, "share": contrib / vol if vol else contrib},
        index=cov.index,
    )
    return out[out["weight"] != 0]


def tracking_error(weights: pd.Series, benchmark: pd.Series, cov: pd.DataFrame) -> float:
    """Ex-ante tracking error of ``weights`` against ``benchmark`` weights.

    A name missing from one side genuinely has zero weight there, so the
    union is filled with zeros (unlike a missing covariance, which is unknown).
    """
    names = weights.index.union(benchmark.index)
    active = weights.reindex(names, fill_value=0.0) - benchmark.reindex(names, fill_value=0.0)
    return portfolio_vol(active, cov)


def parametric_var(vol_annual: float, nav: float, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """Normal (variance-covariance) VaR in currency units, reported as a positive loss.

    Assumes normal returns, so it understates fat-tailed losses; it is a
    monitoring gauge, not a capital number.
    """
    z = NormalDist().inv_cdf(confidence)
    return z * vol_annual * np.sqrt(horizon_days / TRADING_DAYS) * nav


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative return path (negative number)."""
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())
