"""Factor exposures: what actually drives a portfolio's returns.

``fit_factor_model`` regresses a return series on factor returns by OLS and
reports Newey-West (HAC) t-statistics, because daily financial returns are
heteroskedastic and mildly autocorrelated, which makes plain OLS standard
errors overconfident.

``variance_decomposition`` then splits the series' variance into each
factor's share plus a stock-specific remainder. In-sample, OLS residuals are
orthogonal to the factors, so the shares add up to the series' total variance
exactly; the tests check that identity rather than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import TRADING_DAYS


@dataclass(frozen=True)
class FactorFit:
    alpha_annual: float
    betas: pd.Series
    t_stats: pd.Series  # Newey-West, includes "alpha"
    r_squared: float
    residual_vol_annual: float
    n_obs: int
    factor_cov: pd.DataFrame  # daily, on the fitted sample
    residual_var: float  # daily, on the fitted sample


def newey_west_lags(n_obs: int) -> int:
    """Newey & West (1994) plug-in lag length."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def fit_factor_model(y: pd.Series, factors: pd.DataFrame, lags: int | None = None) -> FactorFit:
    data = pd.concat([y.rename("__y__"), factors], axis=1, join="inner").dropna()
    n, k = len(data), factors.shape[1] + 1
    if n < k + 30:
        raise ValueError(f"{n} observations is too few to fit {k - 1} factors")

    Y = data["__y__"].to_numpy()
    F = data[factors.columns].to_numpy()
    X = np.column_stack([np.ones(n), F])
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef

    # Newey-West HAC covariance of the coefficients, Bartlett kernel.
    lags = newey_west_lags(n) if lags is None else lags
    xtx_inv = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    cov_coef = xtx_inv @ meat @ xtx_inv * n / (n - k)  # small-sample correction
    t_stats = coef / np.sqrt(np.diag(cov_coef))

    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    names = list(factors.columns)
    return FactorFit(
        alpha_annual=float(coef[0] * TRADING_DAYS),
        betas=pd.Series(coef[1:], index=names),
        t_stats=pd.Series(t_stats, index=["alpha", *names]),
        r_squared=1 - float(resid @ resid) / ss_tot,
        residual_vol_annual=float(np.std(resid, ddof=1) * np.sqrt(TRADING_DAYS)),
        n_obs=n,
        factor_cov=pd.DataFrame(np.atleast_2d(np.cov(F, rowvar=False, ddof=1)), index=names, columns=names),
        residual_var=float(np.var(resid, ddof=1)),
    )


def variance_decomposition(fit: FactorFit) -> pd.Series:
    """Share of variance from each factor (Euler split of b' S b) and 'specific'.

    A factor's share can be negative when it hedges another factor.
    """
    b = fit.betas.to_numpy()
    s_b = fit.factor_cov.to_numpy() @ b
    total = float(b @ s_b) + fit.residual_var
    shares = pd.Series(b * s_b / total, index=fit.betas.index)
    shares["specific"] = fit.residual_var / total
    return shares


def explained_variance_total(fit: FactorFit) -> float:
    """b' S b + residual variance: equals the series' sample variance in-sample."""
    b = fit.betas.to_numpy()
    return float(b @ fit.factor_cov.to_numpy() @ b) + fit.residual_var
