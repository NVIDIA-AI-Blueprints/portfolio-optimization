# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The blog figures of the simulator comparison in experiments/: which arms they
draw, the readings they compute, and the files they write, on hand-built runs
and on a small scenario comparison, rebalancing backtest, and CPU-against-GPU
experiment with the fake JointFM client and device stand-ins of conftest.py."""

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison import blog as blog_figures, devices_report
from experiments.simulator_comparison.blog import (
    BLOG_RANKING_SCORES,
    BLOG_SKILL_SCORES,
    CHANGE_READING,
    PERFORMANCE_METRICS,
    RISK_FREE_ROW,
    SIZE_READING,
    _dot_groups,
    blog_styles,
    growth_table,
    performance_table,
    risk_free_return,
    risk_return_table,
    size_change_frame,
    universe_size_table,
    write_blog,
)
from experiments.simulator_comparison.config import DevicesConfig, ExperimentConfig
from experiments.simulator_comparison.devices_report import device_tables
from experiments.simulator_comparison.rebalancing import (
    METRICS,
    run_rebalancing_experiment,
)
from experiments.simulator_comparison.rebalancing_report import per_trial_frame
from experiments.simulator_comparison.runner import run_experiment
from experiments.simulator_comparison.scores import PortfolioScores, ScenarioScores
from experiments.simulator_comparison.table import read_results
from experiments.simulator_comparison.universe_sizes import (
    run_universe_size_experiment,
)

ASSETS = 4
STEP_DAYS = 5
FIGURES = (
    "ranking",
    "skill_horizons",
    "growth",
    "risk_return",
    "performance",
    "universe_sizes",
    "runtime",
    "runtime_devices",
    "solvers",
)
TABLES = FIGURES


def _per_trial(arms: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """A per-trial table of two trials of ``arms`` (name to fit type and device)."""
    rows = [
        {"trial": trial, "arm": arm, "fit_type": fit_type, "device": device}
        for trial in range(2)
        for arm, (fit_type, device) in arms.items()
    ]
    return pd.DataFrame(rows).set_index(["trial", "arm"])


def test_one_arm_per_simulator_under_its_blog_name():
    per_trial = _per_trial(
        {
            "jointfm-gpu": ("jointfm", "GPU"),
            "kde-gpu": ("kde", "GPU"),
            "kde-cpu": ("kde", "CPU"),
            "swap-jointfm-window": ("swap-jointfm-window", "CPU"),
            "window-independent": ("window-independent", "CPU"),
            "no_fit-cpu": ("no_fit", "CPU"),
        }
    )
    styles = blog_styles(per_trial)
    assert [style.name for style in styles] == ["no_fit-cpu", "kde-gpu", "jointfm-gpu"]
    assert [style.inline_label for style in styles] == ["Historical", "KDE", "JointFM"]
    assert all(style.filled for style in styles)


def test_the_window_arm_is_required():
    with pytest.raises(ValueError, match="no_fit-cpu"):
        blog_styles(_per_trial({"jointfm-gpu": ("jointfm", "GPU")}))


def _runs(step_returns: dict[str, list[list[float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trial": trial, "arm": arm, "step_returns": path}
            for arm, paths in step_returns.items()
            for trial, path in enumerate(paths)
        ]
    )


def _sized_runs(slopes: dict[str, list[float]], sizes: list[int]) -> pd.DataFrame:
    """Runs whose every reading is trial-specific noise plus ``slopes[arm][trial]`` per asset."""
    arms = {"no_fit-cpu": ("no_fit", "CPU"), "jointfm-gpu": ("jointfm", "GPU")}
    rng = np.random.default_rng(0)
    rows = []
    for arm, per_trial in slopes.items():
        for trial, slope in enumerate(per_trial):
            offset = rng.normal()
            for size in sizes:
                level = offset + slope * size
                rows.append(
                    {"trial": trial, "arm": arm, "size": size}
                    | {metric: level for metric in METRICS}
                    | dict(zip(("fit_type", "device"), arms[arm]))
                )
    return pd.DataFrame(rows)


def test_size_change_is_the_fitted_slope_over_the_span_of_sizes():
    """A reading linear in the size changes by its slope times the span, whatever
    the spacing of the sizes and the trial's own level."""
    sizes = [10, 20, 50, 100]
    runs = _sized_runs(
        {"no_fit-cpu": [0.0, 0.001], "jointfm-gpu": [0.002, 0.003]}, sizes
    )
    changes = size_change_frame(runs, sizes)
    np.testing.assert_allclose(
        changes.loc[
            [
                (0, "no_fit-cpu"),
                (1, "no_fit-cpu"),
                (0, "jointfm-gpu"),
                (1, "jointfm-gpu"),
            ],
            "total_return",
        ],
        [0.0, 0.09, 0.18, 0.27],
        atol=1e-12,
    )
    assert changes.loc[(0, "jointfm-gpu"), "fit_type"] == "jointfm"


def test_size_change_refuses_a_trial_missing_a_size():
    sizes = [10, 20, 30]
    runs = _sized_runs({"no_fit-cpu": [0.0], "jointfm-gpu": [0.001]}, sizes)
    runs = runs[~((runs["arm"] == "jointfm-gpu") & (runs["size"] == 20))]
    with pytest.raises(ValueError, match="each of the sizes"):
        size_change_frame(runs, sizes)


def test_universe_size_table_adds_the_change_against_the_window():
    """Change rows carry no size; JointFM's lead grows by its extra slope times the span."""
    sizes = [10, 40, 100]
    runs = _sized_runs(
        {"no_fit-cpu": [0.0, 0.0, 0.0], "jointfm-gpu": [0.001, 0.001, 0.001]}, sizes
    )
    styles = blog_styles(per_trial_frame(runs[runs["size"] == sizes[0]]))
    table = universe_size_table(runs, styles, sizes)
    at_sizes = table[table["reading"] == SIZE_READING]
    assert sorted(at_sizes["size"].unique()) == sizes
    change = table[table["reading"] == CHANGE_READING].set_index(["metric", "arm"])
    assert change["size"].isna().all()
    jointfm = change.loc[("total_return", "jointfm-gpu")]
    assert jointfm["mean"] == pytest.approx(0.09)
    assert jointfm["difference"] == pytest.approx(0.09)
    assert jointfm["trials_better"] == 3
    # A deepening drawdown is worse, so a larger change of it is no win.
    assert change.loc[("max_drawdown", "jointfm-gpu"), "trials_better"] == 0


def test_growth_compounds_steps_and_reads_drawdown_at_step_ends():
    runs = _runs({"jointfm-gpu": [[0.1, -0.5, 0.2], [0.0, 0.0, 0.0]]})
    styles = blog_styles(
        _per_trial({"no_fit-cpu": ("no_fit", "CPU"), "jointfm-gpu": ("jointfm", "GPU")})
    )
    growth = growth_table(runs, styles[1:], STEP_DAYS)
    # Values 1, 1.1, 0.55, 0.66 and 1, 1, 1, 1; drawdowns 0, 0, 0.5, 0.4 and 0.
    np.testing.assert_allclose(growth["cumulative_return"], [0, 0.05, -0.225, -0.17])
    np.testing.assert_allclose(growth["drawdown"], [0, 0, 0.25, 0.2])
    assert list(growth["trading_day"]) == [0, 5, 10, 15]
    assert set(growth["trials"]) == {2}


def test_growth_refuses_runs_of_different_lengths():
    runs = _runs({"no_fit-cpu": [[0.1, 0.0], [0.1]]})
    styles = blog_styles(_per_trial({"no_fit-cpu": ("no_fit", "CPU")}))
    with pytest.raises(ValueError, match="different step counts"):
        growth_table(runs, styles, STEP_DAYS)


def test_performance_counts_trials_better_in_each_readings_direction():
    """Lower volatility and higher return are better; a tie is not."""
    readings = {
        "no_fit-cpu": {
            "annualized_volatility": [0.3] * 4,
            "annualized_return": [0.05] * 4,
        },
        "jointfm-gpu": {
            "annualized_volatility": [0.2, 0.2, 0.4, 0.2],
            "annualized_return": [0.1, 0.0, 0.0, 0.0],
        },
    }
    devices = {"no_fit-cpu": ("no_fit", "CPU"), "jointfm-gpu": ("jointfm", "GPU")}
    frame = pd.DataFrame(
        [
            {
                **dict.fromkeys(METRICS, 0.5),
                **{metric: values[trial] for metric, values in readings[arm].items()},
                "trial": trial,
                "arm": arm,
                "fit_type": devices[arm][0],
                "device": devices[arm][1],
            }
            for trial in range(4)
            for arm in readings
        ]
    )
    per_trial = per_trial_frame(frame)
    table = performance_table(
        per_trial, blog_styles(per_trial), PERFORMANCE_METRICS
    ).set_index(["metric", "arm"])
    jointfm = table.xs("jointfm-gpu", level="arm")
    assert jointfm.loc["annualized_volatility", "trials_better"] == 3
    assert jointfm.loc["annualized_return", "trials_better"] == 1
    assert jointfm.loc["sharpe_ratio", "trials_better"] == 0
    assert jointfm.loc["annualized_volatility", "mean"] == pytest.approx(0.25)
    assert jointfm.loc["annualized_volatility", "difference"] == pytest.approx(-0.1)
    reference = table.xs("no_fit-cpu", level="arm")
    assert reference[["difference", "trials_better"]].isna().all(axis=None)


def test_the_frontier_runs_through_the_steepest_line_from_cash():
    """The lowest-volatility arm is not the frontier when another rises faster above cash."""
    means = {
        "no_fit-cpu": (0.30, 0.07),
        "gaussian-cpu": (0.10, 0.03),
        "jointfm-gpu": (0.20, 0.09),
    }
    performance = pd.DataFrame(
        [
            {"metric": metric, "arm": arm, "mean": value, "trials": 5}
            for arm, levels in means.items()
            for metric, value in zip(
                ("annualized_volatility", "annualized_return"), levels
            )
        ]
    )
    styles = blog_styles(
        _per_trial(
            {
                "no_fit-cpu": ("no_fit", "CPU"),
                "gaussian-cpu": ("gaussian", "CPU"),
                "jointfm-gpu": ("jointfm", "GPU"),
            }
        )
    )
    table = risk_return_table(performance, styles, 0.01).set_index("arm")
    np.testing.assert_allclose(
        table.loc[list(means), "excess_per_volatility"], [0.2, 0.2, 0.4]
    )
    assert list(table.index[table["frontier"]]) == ["jointfm-gpu"]
    cash = table.loc[RISK_FREE_ROW]
    assert (cash["annualized_volatility"], cash["annualized_return"]) == (0.0, 0.01)
    assert not cash["frontier"]
    assert set(table["trials"]) == {5}


def test_dots_that_would_hide_each_other_share_a_group_in_row_order():
    """The full run's Historical and KDE means, 0.03 points apart, draw as one place."""
    points = pd.DataFrame(
        {"x": [37.31, 38.11, 37.34, 32.96, 0.0], "y": [10.88, 9.44, 10.90, 17.22, 1.1]},
        index=["no_fit", "gaussian", "kde", "jointfm", RISK_FREE_ROW],
    )
    groups = _dot_groups(points, (-1.3, 42.6), (-1.3, 24.2))
    assert groups == [["no_fit", "kde"], ["gaussian"], ["jointfm"], [RISK_FREE_ROW]]


def test_cash_compounds_each_trials_rates_to_an_annual_return():
    """A window at a constant rate earns that rate; the trials' returns are averaged."""
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rates = pd.Series(0.02, index=pd.date_range("2019-12-02", dates[-1], freq="B"))
    rates[rates.index > dates[29]] = 0.04
    runs = pd.DataFrame(
        {
            "trial": [0, 0, 1],
            "trading_start": [dates[0], dates[0], dates[30]],
            "trading_end": [dates[29], dates[29], dates[59]],
        }
    )
    assert risk_free_return(runs, dates, rates) == pytest.approx(0.03)
    runs.loc[1, "trading_end"] = dates[28]
    with pytest.raises(ValueError, match="different windows"):
        risk_free_return(runs, dates, rates)


@pytest.fixture(scope="module")
def blog(tmp_path_factory, module_fake_jointfm, device_sessions):
    """A small comparison and backtest of the window, Gaussian, and fake JointFM
    arms, with the blog written from them once per module."""
    # The fake service's draws scatter around the last log price.
    module_fake_jointfm.envelope.n_input = 1000
    module_fake_jointfm.sample_scale = 0.01
    directory = tmp_path_factory.mktemp("blog")
    rows = 320
    rng = np.random.default_rng(3)
    returns = 0.015 * (rng.normal(size=(rows, 1)) + rng.normal(size=(rows, ASSETS + 2)))
    dates = pd.date_range("2019-01-01", periods=rows, freq="B")
    prices_path = directory / "prices.csv"
    pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f"A{i}" for i in range(ASSETS + 2)],
    ).to_csv(prices_path)
    rates_path = directory / "rates.csv"
    pd.Series(
        0.02, index=pd.date_range("2018-12-03", "2020-06-30", freq="B"), name="rate"
    ).rename_axis("Date").to_csv(rates_path)
    arms = ("no_fit-cpu", "gaussian-cpu", "jointfm")
    config = ExperimentConfig(
        prices=prices_path,
        output=directory / "results.parquet",
        trials=4,
        origins=3,
        assets=ASSETS,
        look_back=64,
        origin_history=64,
        forward=8,
        num_scen=100,
        seed=5,
        arms=arms,
        portfolio_sizes=(1, ASSETS),
        horizons=(1, 3),
        rebalancing=dict(
            output=directory / "rebalancing.parquet",
            scenario_results=directory / "results.parquet",
            arms=arms,
            trials=2,
            trading_days=2 * STEP_DAYS,
            look_forward_window=STEP_DAYS,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=rates_path,
            solver="CLARABEL",
            cvar=dict(
                w_min=0.0,
                w_max=0.6,
                c_min=0.0,
                c_max=0.4,
                L_tar=1.6,
                T_tar=1.0,
                confidence=0.95,
                risk_aversion=1.0,
            ),
        ),
        universe_sizes=dict(
            output=directory / "universe_sizes.parquet",
            arms=arms,
            sizes=(2, ASSETS),
        ),
    )
    portfolios = PortfolioScores(
        config.portfolio_sizes, config.rebalancing.cvar, config.assets
    )
    run_experiment(config, [ScenarioScores(), portfolios])
    run_rebalancing_experiment(config)
    run_universe_size_experiment(config)
    return config, write_blog(config, _device_results(config, device_sessions))


def _device_results(config, device_sessions):
    """Both sessions of a CPU-against-GPU experiment on the backtest of ``config``,
    as task experiments:devices runs them; returns the results file."""
    devices = DevicesConfig(
        experiment=config.output.with_name("experiment.yaml"),
        output=config.output.with_name("devices.parquet"),
        trials=config.rebalancing.trials,
        num_scen=50,
        gpu_solver="CUOPT",
        cpu_solver="CLARABEL",
        paired_solvers=("CUOPT", "CLARABEL"),
        generators=("no_fit-cpu",),
    )
    return device_sessions(devices, config)


def test_writes_only_the_blog_figures_and_their_tables(blog):
    config, outputs = blog
    expected = {
        *(f"{name}.{suffix}" for name in FIGURES for suffix in ("png", "svg")),
        *(f"{name}.csv" for name in TABLES),
    }
    assert config.blog_dir == config.output.with_name("results_blog")
    assert {path.name for path in outputs.files} == expected
    assert {path.name for path in config.blog_dir.iterdir()} == expected
    assert outputs.excluded_trials == []


def test_scenario_tables_hold_the_blog_scores_of_every_simulator(blog):
    config, _ = blog
    simulators = {"no_fit-cpu", "gaussian-cpu", "jointfm-cpu"}
    ranking = pd.read_csv(config.blog_dir / "ranking.csv")
    assert set(ranking["score"]) == set(BLOG_RANKING_SCORES)
    assert set(ranking["arm"]) == simulators
    assert set(ranking["trials"]) == {config.trials}
    for _, won in ranking.groupby("score")["share_won"]:
        assert won.sum() >= 1
    skills = pd.read_csv(config.blog_dir / "skill_horizons.csv")
    assert set(skills["score"]) == set(BLOG_SKILL_SCORES)
    assert set(skills["horizon"]) == set(config.horizons)
    assert set(skills["arm"]) == simulators - {"no_fit-cpu"}


def test_growth_ends_at_each_simulators_mean_total_return(blog):
    """The path compounds the step returns, which span the whole trading window."""
    config, _ = blog
    runs, _ = read_results(config.rebalancing.output)
    growth = pd.read_csv(config.blog_dir / "growth.csv")
    final = growth.groupby("arm")["cumulative_return"].last()
    expected = runs.groupby("arm")["total_return"].mean()
    pd.testing.assert_series_equal(final, expected[final.index], check_names=False)
    assert set(growth.groupby("arm")["step"].max()) == {2}


def test_performance_levels_are_the_means_over_runs(blog):
    config, _ = blog
    runs, _ = read_results(config.rebalancing.output)
    table = pd.read_csv(config.blog_dir / "performance.csv")
    expected = runs.groupby("arm")["max_drawdown"].mean()
    levels = table[table["metric"] == "max_drawdown"].set_index("arm")["mean"]
    pd.testing.assert_series_equal(
        levels.sort_index(), expected.sort_index(), check_names=False
    )


def test_universe_sizes_read_every_size_and_equal_the_backtest_at_full_size(blog):
    """The largest size trades the rebalancing backtest's universe, so its levels
    are the performance figure's."""
    config, _ = blog
    table = pd.read_csv(config.blog_dir / "universe_sizes.csv")
    assert set(table["reading"]) == {SIZE_READING, CHANGE_READING}
    assert table.loc[table["reading"] == CHANGE_READING, "size"].isna().all()
    table = table[table["reading"] == SIZE_READING]
    assert sorted(table["size"].unique()) == list(config.universe_sizes.sizes)
    assert set(table["metric"]) == {"total_return", "max_drawdown"}
    assert set(table["arm"]) == {"no_fit-cpu", "gaussian-cpu", "jointfm-cpu"}
    performance = pd.read_csv(config.blog_dir / "performance.csv")
    at_full = table[(table["size"] == ASSETS) & (table["metric"] == "max_drawdown")]
    expected = performance[performance["metric"] == "max_drawdown"]
    pd.testing.assert_series_equal(
        at_full.set_index("arm")["mean"].sort_index(),
        expected.set_index("arm")["mean"].sort_index(),
    )


def test_risk_return_draws_the_performance_means_and_cash_at_its_rate(blog):
    config, _ = blog
    table = pd.read_csv(config.blog_dir / "risk_return.csv").set_index("arm")
    performance = pd.read_csv(config.blog_dir / "performance.csv")
    for metric in ("annualized_volatility", "annualized_return"):
        means = performance[performance["metric"] == metric].set_index("arm")["mean"]
        pd.testing.assert_series_equal(
            table.loc[means.index, metric], means, check_names=False
        )
    # The fixture's cash earns a constant 2 % a year.
    assert table.loc[RISK_FREE_ROW, "annualized_return"] == pytest.approx(0.02)
    simulators = table.drop(index=RISK_FREE_ROW)
    assert simulators["frontier"].sum() == 1
    assert simulators["excess_per_volatility"].idxmax() == (
        simulators.index[simulators["frontier"]].item()
    )


def test_the_speed_figures_draw_the_device_reports_tables(blog):
    """The blog's speed numbers equal the CPU-against-GPU report's."""
    config, _ = blog
    tables = device_tables(config.output.with_name("devices.parquet"))
    for name, table in tables.items():
        pd.testing.assert_frame_equal(
            # A ratio row leaves its device empty, which is no missing value.
            pd.read_csv(config.blog_dir / f"{name}.csv", keep_default_na=False),
            table,
            check_dtype=False,
        )


def test_no_blog_figure_draws_a_description_below_its_axes(blog, monkeypatch):
    """A figure's description lives in the post's caption, never below its axes."""
    config, _ = blog
    figures = {}

    def record(figure, directory, name):
        figures[name] = figure
        return []

    monkeypatch.setattr(blog_figures, "_save", record)
    monkeypatch.setattr(devices_report, "_save", record)
    write_blog(config, config.output.with_name("devices.parquet"))
    assert set(figures) == set(FIGURES)
    assert {name: figure.get_supxlabel() for name, figure in figures.items()} == (
        dict.fromkeys(FIGURES, "")
    )


def test_a_rate_file_other_than_the_backtests_is_refused(blog, tmp_path):
    config, _ = blog
    other = tmp_path / "rates.csv"
    rates = pd.read_csv(config.rebalancing.risk_free_rates, index_col=0)
    (rates + 0.01).to_csv(other)
    changed = config.model_copy(
        update={
            "rebalancing": config.rebalancing.model_copy(
                update={"risk_free_rates": other}
            )
        }
    )
    with pytest.raises(ValueError, match="risk-free rate file"):
        write_blog(changed, config.output.with_name("devices.parquet"))
