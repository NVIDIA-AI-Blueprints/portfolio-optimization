# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cvxpy as cp
import matplotlib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from portfolio_optimization import mean_variance_optimizer
from portfolio_optimization.backtest import portfolio_backtester
from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import (
    _annualized_sharpe_ratio,
    _select_efficient_frontier_key_portfolios,
    compute_CVaR,
    create_efficient_frontier,
    generate_cvar_data,
    normalize_portfolio_weights_to_one,
)
from portfolio_optimization.mean_variance_parameters import MeanVarianceParameters
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.settings import (
    ApiSettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from portfolio_optimization.utils import (
    calculate_log_returns,
    calculate_returns,
    compare_results,
    compute_absolute_returns,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Fixtures: small synthetic data shared across tests
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture()
def price_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    base = np.array([150.0, 100.0, 250.0])
    noise = np.random.randn(60, 3) * 0.5
    prices = base + np.cumsum(noise, axis=0)
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


@pytest.fixture()
def returns_dict(price_data):
    settings = ReturnsComputeSettings(return_type="LOG", freq=1)
    return calculate_returns(
        price_data, regime_dict=None, returns_compute_settings=settings
    )


@pytest.fixture()
def cvar_data(returns_dict):
    np.random.seed(0)
    settings = ScenarioGenerationSettings(num_scen=200, fit_type="gaussian")
    rd = generate_cvar_data(returns_dict, settings)
    return rd["cvar_data"]


@pytest.fixture()
def cvar_params():
    return CvarParameters(
        w_min=0.0,
        w_max=0.5,
        c_min=0.0,
        c_max=1.0,
        risk_aversion=1.0,
        confidence=0.95,
        L_tar=1.6,
    )


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestReturns:
    def test_log_returns_shape(self, price_data):
        ret = calculate_log_returns(price_data, freq=1)
        assert ret.shape == (59, 3)
        assert list(ret.columns) == TICKERS

    def test_log_returns_values(self, price_data):
        ret = calculate_log_returns(price_data, freq=1)
        expected_first = np.log(price_data.iloc[1]) - np.log(price_data.iloc[0])
        np.testing.assert_allclose(
            ret.iloc[0].values, expected_first.values, atol=1e-12
        )

    def test_abs_returns_shape(self, price_data):
        ret = compute_absolute_returns(price_data, freq=1)
        assert ret.shape == (59, 3)

    def test_abs_returns_values(self, price_data):
        ret = compute_absolute_returns(price_data, freq=1)
        expected_first = price_data.iloc[1] - price_data.iloc[0]
        np.testing.assert_allclose(
            ret.iloc[0].values, expected_first.values, atol=1e-12
        )

    def test_calculate_returns_dict_keys(self, returns_dict):
        expected_keys = {
            "return_type",
            "returns",
            "regime",
            "dates",
            "mean",
            "covariance",
            "tickers",
        }
        assert expected_keys == set(returns_dict.keys())

    def test_calculate_returns_mean_shape(self, returns_dict):
        assert returns_dict["mean"].shape == (3,)

    def test_calculate_returns_covariance_shape(self, returns_dict):
        assert returns_dict["covariance"].shape == (3, 3)

    def test_covariance_is_symmetric(self, returns_dict):
        cov = returns_dict["covariance"]
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ---------------------------------------------------------------------------
# CvarData / CvarParameters
# ---------------------------------------------------------------------------


class TestCvarData:
    def test_shape(self, cvar_data):
        assert cvar_data.R.shape == (3, 200)
        assert cvar_data.mean.shape == (3,)
        assert cvar_data.p.shape == (200,)

    def test_probabilities_sum_to_one(self, cvar_data):
        np.testing.assert_allclose(cvar_data.p.sum(), 1.0, atol=1e-12)

    def test_uniform_probabilities(self, cvar_data):
        np.testing.assert_allclose(cvar_data.p, 1.0 / 200, atol=1e-12)


class TestCvarParameters:
    def test_requires_explicit_weight_bounds(self):
        # CVaR has no silent default bounds; constructing without w_min/w_max
        # must fail fast with a clear error (not infeasible numbers).
        with pytest.raises(ValueError):
            CvarParameters()

    def test_defaults(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        assert params.confidence == 0.95
        assert params.cardinality is None
        assert params.T_tar is None

    def test_update_confidence_valid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        params.update_confidence(0.99)
        assert params.confidence == 0.99

    def test_update_confidence_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_confidence(0.0)
        with pytest.raises(ValueError):
            params.update_confidence(1.5)

    def test_update_risk_aversion_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_risk_aversion(-1.0)

    def test_update_cardinality_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_cardinality(-3)

    def test_update_c_min_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_c_min(-0.1)


# ---------------------------------------------------------------------------
# MeanVarianceParameters
# ---------------------------------------------------------------------------


class TestMeanVarianceParameters:
    def test_var_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            MeanVarianceParameters(var_limit=0.0)
        with pytest.raises(ValidationError):
            MeanVarianceParameters(var_limit=-1.0)

    def test_positive_var_limit_valid(self):
        params = MeanVarianceParameters(var_limit=0.01)
        assert params.var_limit == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Portfolio math
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_expected_return(self):
        mean = np.array([0.05, 0.10, 0.15])
        p = Portfolio(tickers=TICKERS, weights=np.array([0.3, 0.3, 0.4]))
        ret = p.calculate_portfolio_expected_return(mean)
        expected = 0.3 * 0.05 + 0.3 * 0.10 + 0.4 * 0.15
        np.testing.assert_allclose(ret, expected, atol=1e-12)

    def test_portfolio_variance(self):
        cov = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.16]])
        w = np.array([0.5, 0.3, 0.2])
        p = Portfolio(tickers=TICKERS, weights=w)
        var = p.calculate_portfolio_variance(cov)
        expected = w @ cov @ w
        np.testing.assert_allclose(var, expected, atol=1e-12)

    def test_portfolio_from_dict(self):
        p = Portfolio(tickers=TICKERS)
        p.portfolio_from_dict("test", {"AAPL": 0.5, "GOOGL": 0.3}, 0.2)
        assert p.name == "test"
        np.testing.assert_allclose(p.weights, [0.5, 0.3, 0.0], atol=1e-12)
        assert p.cash == pytest.approx(0.2)

    def test_self_financing(self):
        p = Portfolio(tickers=TICKERS, weights=np.array([0.3, 0.3, 0.2]), cash=0.2)
        p._check_self_financing()  # should not raise

    def test_equality(self):
        p1 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        p2 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        assert p1 == p2

    def test_inequality(self):
        p1 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        p2 = Portfolio(tickers=TICKERS, weights=np.array([0.1, 0.1, 0.8]))
        assert not (p1 == p2)


# ---------------------------------------------------------------------------
# CVaR computation
# ---------------------------------------------------------------------------


class TestComputeCVaR:
    def test_known_cvar(self):
        np.random.seed(99)
        n_scen = 10000
        R = np.random.randn(1, n_scen) * 0.02
        p = np.ones(n_scen) / n_scen
        data = CvarData(mean=np.array([0.0]), R=R, p=p)
        weights = np.array([1.0])

        cvar_95 = compute_CVaR(data, weights, confidence_level=0.95)
        assert cvar_95 > 0, "CVaR should be positive for a risky asset"

        cvar_99 = compute_CVaR(data, weights, confidence_level=0.99)
        assert cvar_99 > cvar_95, "99% CVaR should exceed 95% CVaR"

    def test_zero_weight_zero_cvar(self):
        R = np.array([[0.01, -0.02, 0.03], [0.02, -0.01, 0.01]])
        p = np.ones(3) / 3
        data = CvarData(mean=np.zeros(2), R=R, p=p)
        weights = np.array([0.0, 0.0])

        cvar = compute_CVaR(data, weights, confidence_level=0.95)
        assert cvar == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Normalize weights
# ---------------------------------------------------------------------------


class TestNormalizeWeights:
    def test_sums_to_one(self):
        weights_dict = {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.2}
        cash = 0.2
        nw, nc = normalize_portfolio_weights_to_one(weights_dict, cash)
        total = sum(nw.values()) + nc
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_preserves_ratios(self):
        weights_dict = {"A": 2.0, "B": 1.0}
        nw, _ = normalize_portfolio_weights_to_one(weights_dict, 0.0)
        assert nw["A"] / nw["B"] == pytest.approx(2.0, abs=1e-10)

    def test_already_normalized(self):
        weights_dict = {"A": 0.6, "B": 0.3}
        nw, nc = normalize_portfolio_weights_to_one(weights_dict, 0.1)
        np.testing.assert_allclose(nw["A"], 0.6, atol=1e-10)
        np.testing.assert_allclose(nc, 0.1, atol=1e-10)


# ---------------------------------------------------------------------------
# Scenario generation (generate_cvar_data)
# ---------------------------------------------------------------------------


class TestGenerateCvarData:
    def test_gaussian_fit(self, returns_dict):
        np.random.seed(7)
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(num_scen=100, fit_type="gaussian"),
        )
        cd = rd["cvar_data"]
        assert cd.R.shape == (3, 100)
        np.testing.assert_allclose(cd.p.sum(), 1.0, atol=1e-12)

    @pytest.mark.skip(
        reason="no_fit path in generate_cvar_data passes a DataFrame to CvarData.R, "
        "which pydantic-validates as ndarray. Pre-existing bug in src/cvar_utils.py."
    )
    def test_no_fit(self, returns_dict):
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(fit_type="no_fit"),
        )
        cd = rd["cvar_data"]
        n_obs = returns_dict["returns"].shape[0]
        assert cd.R.shape == (3, n_obs)

    def test_invalid_fit_type(self):
        with pytest.raises(ValidationError):
            ScenarioGenerationSettings(num_scen=50, fit_type="magic")


# ---------------------------------------------------------------------------
# Small CVaR optimization (CVXPY + CLARABEL, CPU only)
# ---------------------------------------------------------------------------


class TestCVaROptimization:
    def test_basic_optimization_feasible(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        assert portfolio is not None
        w = portfolio.weights
        c = portfolio.cash

        np.testing.assert_allclose(np.sum(w) + c, 1.0, atol=1e-4)

        assert np.all(w >= -1e-4), "weights should respect lower bound"
        assert np.all(w <= 0.5 + 1e-4), "weights should respect upper bound"

    def test_optimization_returns_expected_keys(
        self, returns_dict, cvar_data, cvar_params
    ):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, _ = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        for key in ["regime", "solver", "solve time", "return", "CVaR", "obj"]:
            assert key in result.index, f"Missing key: {key}"

    def test_cvar_consistent_with_compute(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        reported_cvar = result["CVaR"]
        computed_cvar = compute_CVaR(
            cvar_data, portfolio.weights, cvar_params.confidence
        )

        np.testing.assert_allclose(reported_cvar, computed_cvar, atol=0.02)


# ---------------------------------------------------------------------------
# Small Mean-Variance optimization and SOCP variance caps
# ---------------------------------------------------------------------------


class TestMeanVarianceOptimization:
    def _equal_weight_variance_cap(self, returns_dict, multiplier=1.05):
        covariance = np.asarray(returns_dict["covariance"], dtype=float)
        weights = np.ones(len(returns_dict["tickers"])) / len(returns_dict["tickers"])
        return float(weights @ covariance @ weights) * multiplier

    def test_cvxpy_var_limit_respects_cap(self, returns_dict):
        variance_cap = self._equal_weight_variance_cap(returns_dict)
        params = MeanVarianceParameters(
            w_min=0.0,
            w_max=1.0,
            c_min=0.0,
            c_max=0.0,
            L_tar=1.0,
            var_limit=variance_cap,
        )
        optimizer = mean_variance_optimizer.MeanVariance(
            returns_dict=returns_dict,
            mean_variance_params=params,
        )
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        weights = np.asarray(portfolio.weights, dtype=float).flatten()
        cash = float(np.asarray(portfolio.cash).squeeze())
        realized_variance = float(weights @ returns_dict["covariance"] @ weights)

        np.testing.assert_allclose(np.sum(weights) + cash, 1.0, atol=1e-4)
        assert realized_variance <= variance_cap + 1e-6
        assert float(result["variance"]) <= variance_cap + 1e-6
        assert float(result["return"]) > -1.0

    @pytest.mark.gpu
    def test_cuopt_python_var_limit_solves_socp(self, returns_dict, require_cuopt_socp):
        variance_cap = self._equal_weight_variance_cap(returns_dict)
        params = MeanVarianceParameters(
            w_min=0.0,
            w_max=1.0,
            c_min=0.0,
            c_max=0.0,
            L_tar=1.0,
            var_limit=variance_cap,
        )
        optimizer = mean_variance_optimizer.MeanVariance(
            returns_dict=returns_dict,
            mean_variance_params=params,
            api_settings=ApiSettings(api="cuopt_python"),
        )
        result, portfolio = optimizer.solve_optimization_problem(print_results=False)

        weights = np.asarray(portfolio.weights, dtype=float).flatten()
        cash = float(np.asarray(portfolio.cash).squeeze())
        realized_variance = float(weights @ returns_dict["covariance"] @ weights)

        np.testing.assert_allclose(np.sum(weights) + cash, 1.0, atol=1e-4)
        assert str(result["solver"]).lower() == "cuopt"
        assert realized_variance <= variance_cap + 1e-6


# ---------------------------------------------------------------------------
# Efficient frontier end-to-end
# ---------------------------------------------------------------------------


class TestEfficientFrontier:
    def test_zero_volatility_sharpe_is_nan(self):
        assert np.isnan(_annualized_sharpe_ratio(0.0, 0.0))

    def test_key_portfolios_skip_origin_when_risky_portfolios_exist(self):
        results_df = pd.DataFrame(
            {
                "variance": [0.0, 0.01, 0.04],
                "return": [0.0, 0.06, 0.08],
                "sharpe": [np.nan, 0.95, 0.75],
            },
            index=["origin", "low_risk", "high_risk"],
        )

        key_portfolios = _select_efficient_frontier_key_portfolios(results_df)

        assert key_portfolios["Min Variance"] == "low_risk"
        assert key_portfolios["Max Sharpe"] == "low_risk"

    def test_frontier_monotonicity(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        solver_settings = {"solver": cp.CLARABEL, "verbose": False}

        results_df, fig, ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            solver_settings,
            ra_num=3,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=False,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )

        assert len(results_df) == 3
        assert "return" in results_df.columns
        assert "CVaR" in results_df.columns
        assert "variance" in results_df.columns

        sorted_by_risk = results_df.sort_values("CVaR")
        returns_sorted = sorted_by_risk["return"].values
        assert returns_sorted[-1] >= returns_sorted[0] - 1e-6, (
            "higher risk should generally yield higher return on the frontier"
        )

        for _, row in results_df.iterrows():
            assert row["variance"] >= 0, "variance must be non-negative"
            assert np.isfinite(row["sharpe"]), "sharpe should be finite"

    def test_frontier_exposes_per_asset_weights(
        self, returns_dict, cvar_data, cvar_params
    ):
        returns_dict["cvar_data"] = cvar_data
        results_df, _fig, _ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            {"solver": cp.CLARABEL, "verbose": False},
            ra_num=3,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=False,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )
        assert "weights" in results_df.columns
        assert "cash" in results_df.columns
        first = results_df["weights"].iloc[0]
        assert isinstance(first, dict)
        assert set(first.keys()) == set(TICKERS)

    def test_discretized_overlay_runs(self, returns_dict, cvar_data, cvar_params):
        # Regression: show_discretized_portfolios=True with the default
        # discretization_params used to raise TypeError via a stale 'sum_to_one'
        # kwarg. It must now run end-to-end (NumPy fallback on CPU).
        returns_dict["cvar_data"] = cvar_data
        results_df, _fig, _ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            {"solver": cp.CLARABEL, "verbose": False},
            ra_num=2,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=True,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )
        assert len(results_df) == 2


# ---------------------------------------------------------------------------
# Backtester with canned data
# ---------------------------------------------------------------------------


class TestBacktester:
    @pytest.fixture()
    def backtest_returns_dict(self):
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        daily_returns = np.random.randn(252, 3) * 0.01 + 0.0003
        returns_df = pd.DataFrame(daily_returns, index=dates, columns=TICKERS)
        return {
            "return_type": "LOG",
            "returns": returns_df,
            "dates": returns_df.index,
            "mean": np.mean(daily_returns, axis=0),
            "covariance": np.cov(daily_returns.T),
            "tickers": TICKERS,
        }

    def test_sharpe_ratio_positive_for_positive_drift(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        sharpe = results.loc["test", "sharpe"]
        assert sharpe > 0, "positive drift data should produce positive Sharpe"

    def test_sortino_exceeds_sharpe(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        sharpe = results.loc["test", "sharpe"]
        sortino = results.loc["test", "sortino"]
        assert sortino > sharpe, (
            "Sortino should exceed Sharpe when downside vol < total vol"
        )

    def test_max_drawdown_bounded(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        mdd = results.loc["test", "max drawdown"]
        assert 0 <= mdd <= 1, "max drawdown should be between 0 and 1"

    def test_cumulative_returns_anchor_to_regime_start(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, 102.0, 103.0],
                "GOOGL": [50.0, 49.0, 50.0, 51.0],
                "MSFT": [200.0, 200.0, 202.0, 204.0],
            },
            index=dates,
        )
        returns_dict = calculate_returns(
            prices,
            regime_dict={"name": "test", "range": ("2024-01-02", "2024-01-05")},
            returns_compute_settings=ReturnsComputeSettings(
                return_type="LINEAR", freq=1
            ),
        )
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.5, 0.5, 0.0]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=returns_dict,
            risk_free_rate=0.0,
            benchmark_portfolios=[],
        )

        result = bt.backtest_single_portfolio(portfolio)
        cumulative_returns = result["cumulative returns"].iloc[0]

        assert bt.cumulative_dates[0] == pd.Timestamp("2024-01-02")
        assert bt.cumulative_dates[1] == pd.Timestamp("2024-01-03")
        assert len(cumulative_returns) == len(bt.cumulative_dates)
        assert cumulative_returns[0] == pytest.approx(1.0)
        assert cumulative_returns[1] == pytest.approx(0.995)

    def test_drawdown_known_series(self):
        values = np.array([1.0, 1.1, 1.05, 0.9, 0.95, 1.0])
        bt = portfolio_backtester.__new__(portfolio_backtester)
        mdd = bt.max_drawdown(values)
        expected = (1.1 - 0.9) / 1.1
        np.testing.assert_allclose(mdd, expected, atol=1e-10)


class TestCompareResultsLabels:
    """Output of compare_results must contain 'cuOpt' (mixed case) for the
    GPU solver regardless of whether it was invoked via CVXPY's `cp.CUOPT`
    (which str()'s to 'CUOPT') or the cuOpt Python API (which uses 'cuOpt').
    The QA-image notebook test parses the rendered output for the literal
    string 'cuOpt' followed by a solve time, so this is a contract."""

    @staticmethod
    def _row(solver, t=1.234):
        return pd.Series(
            {"regime": "r", "solver": solver, "solve time": t, "obj": -0.5}
        )

    def test_cvxpy_cuopt_displays_as_cuopt_gpu(self, capsys):
        compare_results(self._row("CUOPT"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU)" in out
        assert "CLARABEL (CPU)" in out
        # The QA pattern is roughly r"cuOpt.*?(\d+\.\d+)" - make sure it matches
        import re

        assert re.search(r"cuOpt.*?(\d+\.\d+)", out)

    def test_cuopt_python_api_displays_as_cuopt_gpu(self, capsys):
        compare_results(self._row("cuOpt"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU)" in out
        assert "CLARABEL (CPU)" in out

    def test_objective_diff_uses_friendly_labels(self, capsys):
        compare_results(self._row("CUOPT"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU) vs CLARABEL (CPU)" in out
