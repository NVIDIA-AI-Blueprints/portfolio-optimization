# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture()
def portfolio_components():
    module = importlib.import_module("portfolio_optimization.portfolio_components")
    plt.close("all")
    return module


@pytest.fixture()
def sample_price_data(tmp_path):
    labels = ("FICO", "LLY", "CAH", "OXY", "STLD")
    dates = pd.to_datetime(
        ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]
    )
    prices = pd.DataFrame(
        [
            [100.0, 200.0, 50.0, 80.0, 40.0],
            [105.0, 198.0, 52.0, 88.0, 44.0],
            [102.0, 210.0, 51.0, 84.0, 48.0],
            [110.0, 220.0, 55.0, 96.0, 50.0],
        ],
        columns=labels,
        index=dates,
    )
    prices.index.name = "Date"
    data_path = tmp_path / "sp500.csv"
    prices.to_csv(data_path)
    return data_path, labels, dates, prices


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_weight_port_oos_balances_long_short_synthetic_example(
    portfolio_components,
):
    labels = ("A1", "A2", "A3", "A4", "A5")
    weights = np.array([-0.2, 0.4, 0.4, 0.2, 0.2])
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    prices = pd.DataFrame(
        [
            [25, 40, 40, 40, 40],
            [50, 40, 40, 80, 40],
        ],
        columns=labels,
        index=dates,
    ).to_numpy()

    portv, returned_dates = portfolio_components.weightPortOOS(
        labels,
        length=2,
        D=len(labels),
        w=weights,
        prices=prices,
        dates=dates,
        start="2022-01-03",
        end="2022-01-04",
    )

    np.testing.assert_allclose(portv, np.array([1.0, 1.0]))
    pd.testing.assert_index_equal(returned_dates, dates)


def test_weight_port_oos_loads_history_when_prices_are_default(
    portfolio_components,
    monkeypatch,
):
    labels = ("A1", "A2")
    weights = np.array([0.25, 0.75])
    dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    prices = np.array(
        [
            [10.0, 20.0],
            [20.0, 10.0],
        ]
    )
    calls = {}

    def fake_get_hist_prices(lab, w, length, start, end):
        calls["lab"] = lab
        calls["w"] = w
        calls["length"] = length
        calls["start"] = start
        calls["end"] = end
        return prices, dates

    monkeypatch.setattr(portfolio_components, "getHistPrices", fake_get_hist_prices)

    portv, returned_dates = portfolio_components.weightPortOOS(
        labels,
        length=2,
        D=len(labels),
        w=weights,
        start="2022-01-03",
        end="2022-01-04",
    )

    assert calls["lab"] == labels
    np.testing.assert_array_equal(calls["w"], weights)
    assert calls["length"] == 2
    assert calls["start"] == "2022-01-03"
    assert calls["end"] == "2022-01-04"
    np.testing.assert_allclose(portv, np.array([1.0, 0.875]))
    pd.testing.assert_index_equal(returned_dates, dates)


def test_get_hist_prices_loads_requested_columns_and_dates(
    portfolio_components,
    sample_price_data,
    monkeypatch,
):
    data_path, labels, dates, prices_df = sample_price_data
    monkeypatch.setattr(portfolio_components, "data_path", str(data_path))
    weights = np.tile(1 / len(labels), len(labels))

    prices, dates = portfolio_components.getHistPrices(
        labels,
        weights,
        length=4,
        start="2022-01-03",
        end="2022-01-06",
    )

    np.testing.assert_allclose(prices, prices_df.to_numpy())
    assert list(dates) == ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]


def test_plot_mult_series_lg_sh_handles_five_stock_history(
    portfolio_components,
    sample_price_data,
):
    _, labels, dates, prices_df = sample_price_data
    weights = np.tile(1 / len(labels), len(labels))
    prices = prices_df.to_numpy()
    portv = np.mean(prices / prices[0, :], axis=1)

    portfolio_components.plotMultSeriesLgSh(
        prices,
        labels,
        weights,
        dates,
        isShow=False,
        isPlotMean=False,
        isShowPortv=True,
        portv=portv,
        linewidth=1,
        main="test",
    )

    axes = plt.gcf().axes
    assert len(axes) == 1
    assert len(axes[0].lines) == len(labels) + 1
    assert [line.get_label() for line in axes[0].lines[: len(labels)]] == list(labels)


def test_display_components_matches_existing_long_short_example(
    portfolio_components,
    sample_price_data,
    monkeypatch,
):
    data_path, labels, _, prices_df = sample_price_data
    monkeypatch.setattr(portfolio_components, "data_path", str(data_path))
    start = "2022-01-03"
    end = "2022-01-06"
    prices = prices_df.to_numpy()

    equal_weights = np.tile(1 / len(labels), len(labels))
    equal_portv = portfolio_components.displayComponents(
        labels,
        equal_weights,
        length=prices.shape[0],
        start=start,
        end=end,
        isShow=False,
        isShowPortv=True,
    )
    np.testing.assert_allclose(equal_portv[0], 1.0)
    assert equal_portv.shape == (4,)

    long_short_weights = np.array([-0.2, 0.4, 0.4, 0.2, 0.2])
    long_short_portv = portfolio_components.displayComponents(
        labels,
        long_short_weights,
        length=prices.shape[0],
        start=start,
        end=end,
        isShow=False,
        isShowPortv=True,
    )
    expected = np.zeros(prices.shape[0])
    for i in range(prices.shape[0]):
        for d in range(prices.shape[1]):
            expected[i] += long_short_weights[d] * prices[i, d] / prices[0, d]

    np.testing.assert_allclose(long_short_portv, expected)
