import numpy as np
import pandas as pd
import pytest

from portfolio_guardrails.factors import (
    explained_variance_total,
    fit_factor_model,
    variance_decomposition,
)
from portfolio_guardrails.risk import (
    ewma_covariance,
    max_drawdown,
    parametric_var,
    portfolio_vol,
    risk_contributions,
    tracking_error,
)

NAMES = ["A", "B", "C", "D"]


def random_cov(seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(4, 4))
    return pd.DataFrame(m @ m.T / 10, index=NAMES, columns=NAMES)


@pytest.mark.parametrize("seed", range(10))
def test_euler_contributions_sum_to_volatility(seed):
    cov = random_cov(seed)
    w = pd.Series(np.random.default_rng(seed + 100).dirichlet(np.ones(4)), index=NAMES)
    rc = risk_contributions(w, cov)
    assert rc["contribution"].sum() == pytest.approx(portfolio_vol(w, cov), rel=1e-12)
    assert rc["share"].sum() == pytest.approx(1.0, rel=1e-12)


def test_unknown_asset_fails_closed_instead_of_counting_as_riskless():
    with pytest.raises(KeyError):
        portfolio_vol(pd.Series({"A": 0.5, "Z": 0.5}), random_cov(0))


def test_tracking_error_zero_for_same_weights_positive_otherwise():
    cov = random_cov(1)
    w = pd.Series([0.25] * 4, index=NAMES)
    assert tracking_error(w, w, cov) == pytest.approx(0.0, abs=1e-15)
    assert tracking_error(w, pd.Series({"A": 1.0}), cov) > 0


def test_ewma_matches_hand_computation():
    r = pd.DataFrame({"x": [0.01, -0.02], "y": [0.03, 0.01]})
    lam = 0.5
    w_old, w_new = lam / (1 + lam), 1 / (1 + lam)  # normalised weights, newest heaviest
    expected_xy = (w_old * 0.01 * 0.03 + w_new * -0.02 * 0.01) * 252
    assert ewma_covariance(r, lam).loc["x", "y"] == pytest.approx(expected_xy)


def test_var_and_drawdown_known_values():
    # 1.6449 * 16% / sqrt(252) * 1,000,000
    assert parametric_var(0.16, 1_000_000) == pytest.approx(16_578.6, abs=1)
    # wealth 1.10 -> 0.55 -> 0.66: worst fall is 0.55 / 1.10 - 1
    assert max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.5)


def planted(seed=1, n=2000, betas=(1.2, -0.4, 0.8), noise=0.004):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(rng.normal(0, 0.01, size=(n, 3)), columns=["MKT", "RATES", "USD"])
    y = pd.Series(0.0001 + f.to_numpy() @ np.array(betas) + rng.normal(0, noise, n))
    return y, f


def test_factor_model_recovers_planted_betas():
    y, f = planted()
    fit = fit_factor_model(y, f)
    assert np.allclose(fit.betas.to_numpy(), [1.2, -0.4, 0.8], atol=0.03)
    assert (fit.t_stats[["MKT", "RATES", "USD"]].abs() > 10).all()


def test_control_scrambled_returns_show_no_exposure():
    y, f = planted()
    scrambled = pd.Series(np.random.default_rng(9).permutation(y.to_numpy()))
    fit = fit_factor_model(scrambled, f)
    assert (fit.betas.abs() < 0.1).all()
    assert (fit.t_stats[["MKT", "RATES", "USD"]].abs() < 3).all()


def test_variance_decomposition_adds_up_to_sample_variance():
    y, f = planted()
    fit = fit_factor_model(y, f)
    assert explained_variance_total(fit) == pytest.approx(y.var(ddof=1), rel=1e-10)
    assert variance_decomposition(fit).sum() == pytest.approx(1.0)


def test_newey_west_widens_errors_when_residuals_are_autocorrelated():
    rng = np.random.default_rng(3)
    n = 3000

    def ar1(phi):
        e = rng.normal(size=n)
        for i in range(1, n):
            e[i] += phi * e[i - 1]
        return e

    f = pd.DataFrame({"F": ar1(0.6) * 0.01})
    y = pd.Series(0.5 * f["F"].to_numpy() + ar1(0.6) * 0.01)
    plain = fit_factor_model(y, f, lags=0)
    hac = fit_factor_model(y, f)
    assert abs(hac.t_stats["F"]) < 0.8 * abs(plain.t_stats["F"])


def test_too_little_history_is_refused():
    y, f = planted(n=30)
    with pytest.raises(ValueError):
        fit_factor_model(y, f)
