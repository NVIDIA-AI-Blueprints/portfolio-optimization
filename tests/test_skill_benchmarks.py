# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 3: portfolio optimization skill performance benchmarks (GPU-gated regression suite).

Runs the documented SKILL.md workflows end-to-end with NVIDIA cuOpt and asserts
each meets the standards in ``tests/benchmarks/thresholds.toml``. The whole
module auto-skips when the cuOpt/cuML GPU runtime is unavailable, so it is
harmless on developer laptops and the no-GPU CI lane.

Run explicitly with:  ``uv run pytest -m gpu``
"""

import pathlib
import sys

import pytest

pytestmark = pytest.mark.gpu

# Skip the whole module cleanly when the GPU runtime is absent.
pytest.importorskip("cuopt", reason="cuOpt GPU runtime required for skill benchmarks")
pytest.importorskip("cuml", reason="cuML GPU runtime required for skill benchmarks")

_BENCHMARKS_DIR = pathlib.Path(__file__).resolve().parent / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS_DIR))

import benchmark_workflows as bw  # noqa: E402


@pytest.fixture(scope="module")
def thresholds():
    return bw.load_thresholds()


@pytest.fixture(scope="module")
def prepared():
    """Load data + scenarios + the optimal portfolio once for the whole module."""
    try:
        prices, _ = bw.load_prices()
    except bw.DataUnavailable as exc:
        pytest.skip(f"price data unavailable: {exc}")
    returns_dict = bw.prepare(prices)
    build_metrics, portfolio = bw.run_build_optimal(returns_dict)
    return {
        "prices": prices,
        "returns_dict": returns_dict,
        "build_metrics": build_metrics,
        "portfolio": portfolio,
    }


def test_build_optimal_is_non_degenerate(prepared, thresholds):
    fails = bw.check_build_optimal(
        prepared["build_metrics"], thresholds["build_optimal"]
    )
    assert not fails, fails


def test_socp_variance_limit_solves(prepared, thresholds, require_cuopt_socp):
    metrics = bw.run_socp_variance_limit(prepared["returns_dict"])
    fails = bw.check_socp_variance_limit(metrics, thresholds["socp_variance_limit"])
    assert not fails, fails


def test_efficient_frontier_is_monotonic(prepared, thresholds):
    metrics = bw.run_efficient_frontier(prepared["returns_dict"])
    fails = bw.check_efficient_frontier(metrics, thresholds["efficient_frontier"])
    assert not fails, fails


def test_weights_table_exposes_per_asset_weights(prepared, thresholds):
    metrics = bw.run_weights_table(prepared["returns_dict"])
    fails = bw.check_weights_table(metrics, thresholds["weights_table"])
    assert not fails, fails


def test_backtest_beats_naive_benchmark(prepared, thresholds):
    metrics = bw.run_backtest(prepared["returns_dict"], prepared["portfolio"])
    fails = bw.check_backtest(metrics, thresholds["backtest"])
    assert not fails, fails


def test_rebalance_produces_schedule(prepared, thresholds):
    try:
        metrics = bw.run_rebalance(prepared["prices"])
    except bw.DataUnavailable as exc:
        pytest.skip(str(exc))
    fails = bw.check_rebalance(metrics, thresholds["rebalance"])
    assert not fails, fails
