# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blog figures of a configuration's results: the readings a post about JointFM shows.

The two reports (``report``, ``rebalancing_report``) draw every reading of
every arm, so they show where JointFM needs to improve as much as where it
leads. This module draws only the readings the blog post builds its story on,
for the scenario simulators, from the scenario and rebalancing results one
configuration names and the results of the CPU-against-GPU experiment
(``devices.py``), into ``<output stem>_blog/`` beside the scenario results
(``ExperimentConfig.blog_dir``). Each figure is a PNG and an SVG, with a CSV
of exactly the numbers it draws.

- ``ranking``: per score of ``BLOG_RANKING_SCORES``, each simulator's share of
  trials with the best next-day score.
- ``skill_horizons``: skill against replaying the window per score of
  ``BLOG_SKILL_SCORES``, over the horizons.
- ``growth``: the rebalanced portfolio's mean cumulative return and the mean of
  each trial's drawdown from its own running peak after every step.
- ``risk_return``: each simulator's mean annualized volatility against its mean
  annualized return, cash at the backtests' mean risk-free return, and the
  frontier with cash: the line from cash through the simulator whose line from
  cash is steepest (``risk_return_table``).
- ``performance``: every reading of ``PERFORMANCE_METRICS``, with the number of
  trials in which each simulator did better than replaying the window.
- ``universe_sizes``: each simulator's mean total return over the universe
  sizes of the nested-subset backtest (``universe_sizes``), JointFM against the
  band of the other simulators; its CSV adds the maximum drawdown and each
  reading's change from the smallest to the largest size
  (``universe_size_table``).
- ``runtime``, ``runtime_devices``, and ``solvers``: the CPU-against-GPU
  experiment's tables (``devices_report``).

Every statistic comes from ``analysis`` and ``rebalancing_report``, so the
numbers equal the reports'. The figures name the
simulators as the post does (``BLOG_NAMES``), and the scenario and
rebalancing figures draw one arm per simulator, so each name appears once:
its GPU arm where it ran on both devices. Titles
describe what is drawn and never state a finding, because the same figures
re-render from every run.
"""

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from portfolio_optimization import risk_free

from .analysis import (
    REFERENCE_ARM,
    aggregate,
    complete_trials,
    ranking,
    skill,
    skill_summary,
    stack_horizons,
)
from .blog_style import (
    BLOG_NAMES,
    TEXT_SIZE,
    _axis_labels,
    _headline,
    _panel_title,
    _style_text,
)
from .config import ExperimentConfig, load_config, load_devices_config
from .devices_report import device_tables, write_device_figures
from .plotting import (
    GRID,
    INK_SECONDARY,
    INTERVAL_MARKER_AREA,
    LINE_WIDTH,
    MARKER_EDGE,
    MUTED,
    SURFACE,
    ArmStyle,
    _curves,
    _figure,
    _interval_panel,
    _save,
    _target_line,
    simulator_styles,
)
from .rebalancing import METRICS
from .rebalancing_report import (
    METRIC_LABELS,
    PERCENT,
    comparisons,
    complete_runs,
    per_trial_frame,
    summary,
)
from .regimes import check_prices
from .report import ReportOutputs, _check_scores_recorded
from .runner import load_prices
from .scores import PORTFOLIO_CRPS
from .table import read_results
from .universe_sizes import complete_universe_runs

NEXT_DAY = 1
# (title, what the score reads) per ranking score; the variogram score that
# ``analysis.RANKING_SCORES`` also ranks by is left to the report.
BLOG_RANKING_SCORES = {
    "energy_score": ("Energy score", "all assets of the universe jointly"),
    PORTFOLIO_CRPS: ("Portfolio CRPS", "random portfolios of the universe's assets"),
}
BLOG_SKILL_SCORES = {
    "energy_score": ("Energy score", "all assets of the universe jointly"),
    "crps": ("CRPS", "each asset on its own"),
}
PERFORMANCE_METRICS = (
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
)
# Axis label of each reading's level; ``METRIC_LABELS`` labels its differences.
LEVEL_UNITS = {
    "annualized_return": "% per year",
    "annualized_volatility": "% per year",
    "sharpe_ratio": "annualized",
    "sortino_ratio": "annualized, downside risk only",
    "max_drawdown": "% below the running peak",
    "calmar_ratio": "annualized return over max drawdown",
}
# Readings of the rebalanced portfolio over the universe sizes: the figure
# draws the first, and the post's text cites both from the CSV.
UNIVERSE_SIZE_METRICS = ("total_return", "max_drawdown")
# ``reading`` of a universe_sizes.csv row: the levels at one size, or the
# change over all sizes (``universe_size_table``).
SIZE_READING = "size"
CHANGE_READING = "change"
# The simulator the post is about; the universe-size figure draws the others
# as one group against it.
FOCUS_FIT_TYPE = "jointfm"
FOCUS_LINE_WIDTH = 3
GROUP_LINE_WIDTH = 1.2
# Opacity of the band between the other simulators' lowest and highest mean.
GROUP_SHADE = 0.15
# Display decimals of a level: percentages to a tenth, ratios to a hundredth.
PERCENT_DECIMALS = 1
RATIO_DECIMALS = 2
# Vertical gap between two end labels, as a share of the axis height.
LABEL_GAP = 0.07
# Room right of the longest horizon for the end labels, as a factor on it.
END_LABEL_ROOM = 2.6
# Steps between two labeled ticks of the growth figure.
GROWTH_TICK_STEPS = 3
# Room right of the widest interval for its value label, as a share of the range.
VALUE_LABEL_ROOM = 0.3
# The row of risk_return.csv that holds cash at the risk-free rate.
RISK_FREE_ROW = "risk_free"
# Room right of the largest volatility, and above and below the drawn returns,
# as a share of the drawn range.
FRONTIER_ROOM = 0.12
# Opacity of the area below the frontier with cash.
FRONTIER_SHADE = 0.1
# Distance within which two risk-return dots hide each other, as a share of
# each axis span.
DOT_OVERLAP = 0.03
# Offset of a dot's label from its center, in points.
DOT_LABEL_OFFSET = 8, 6


@dataclass(frozen=True)
class BlogStyle(ArmStyle):
    """An arm drawn under its simulator's blog name, filled whatever its device."""

    @property
    def filled(self) -> bool:
        return True

    @property
    def inline_label(self) -> str:
        return BLOG_NAMES[self.fit_type]


def blog_styles(per_trial: pd.DataFrame) -> list[BlogStyle]:
    """One style per scenario simulator in simulator order, its GPU arm where it has two.

    Raises
    ------
    ValueError
        If ``REFERENCE_ARM`` is not among the arms, since every blog figure
        reads the simulators against replaying the window.
    """
    # simulator_styles orders CPU before GPU, so a GPU arm replaces its CPU twin.
    by_simulator = {style.fit_type: style for style in simulator_styles(per_trial)}
    styles = [
        BlogStyle(style.name, style.fit_type, style.device)
        for style in by_simulator.values()
    ]
    if REFERENCE_ARM not in {style.name for style in styles}:
        raise ValueError(
            f"The blog reads every simulator against {REFERENCE_ARM}, which the "
            f"results do not contain."
        )
    return styles


def _of_arms(per_trial: pd.DataFrame, styles: Sequence[BlogStyle]) -> pd.DataFrame:
    arms = per_trial.index.get_level_values("arm")
    return per_trial[arms.isin([style.name for style in styles])]


def ranking_table(per_trial: pd.DataFrame, styles: Sequence[BlogStyle]) -> pd.DataFrame:
    """``analysis.ranking`` among the drawn simulators, for ``BLOG_RANKING_SCORES``."""
    drawn = _of_arms(per_trial, styles)
    ranks = ranking(drawn)
    ranks = ranks[ranks["score"].isin(BLOG_RANKING_SCORES)]
    trials = drawn.index.get_level_values("trial").nunique()
    return ranks.assign(trials=trials).reset_index(drop=True)


def skill_table(
    per_trial: dict[int, pd.DataFrame], styles: Sequence[BlogStyle]
) -> pd.DataFrame:
    """Median skill against ``REFERENCE_ARM`` per horizon, score, and drawn simulator."""
    summaries = {
        horizon: skill_summary(skill(_of_arms(table, styles)))
        for horizon, table in per_trial.items()
    }
    stacked = stack_horizons(summaries)
    return stacked[stacked["score"].isin(BLOG_SKILL_SCORES)].reset_index(drop=True)


def growth_table(
    frame: pd.DataFrame, styles: Sequence[BlogStyle], step_days: int
) -> pd.DataFrame:
    """Per drawn simulator and step: mean cumulative return and mean drawdown.

    Each run's value starts at 1 and compounds its ``step_returns``; its
    drawdown after a step is the share its value lies below the highest value
    of the steps up to it, read at step ends only, so it can be shallower than
    the run's ``max_drawdown`` over daily values. Both are averaged over the
    runs, and ``trading_day`` counts ``step_days`` per step.

    Raises
    ------
    ValueError
        If the runs of one simulator cover different numbers of steps.
    """
    tables = []
    for style in styles:
        paths = list(frame.loc[frame["arm"] == style.name, "step_returns"])
        if len({len(path) for path in paths}) != 1:
            raise ValueError(f"The runs of {style.name} cover different step counts.")
        returns = np.array(paths, dtype=float)
        values = np.cumprod(
            np.concatenate([np.zeros((len(returns), 1)), returns], axis=1) + 1, axis=1
        )
        drawdowns = 1 - values / np.maximum.accumulate(values, axis=1)
        steps = np.arange(values.shape[1])
        tables.append(
            pd.DataFrame(
                {
                    "arm": style.name,
                    "step": steps,
                    "trading_day": steps * step_days,
                    "cumulative_return": values.mean(axis=0) - 1,
                    "drawdown": drawdowns.mean(axis=0),
                    "trials": len(values),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def performance_table(
    per_trial: pd.DataFrame, styles: Sequence[BlogStyle], metrics: Sequence[str]
) -> pd.DataFrame:
    """Per drawn simulator and reading of ``metrics``: level and paired difference.

    ``mean``, ``ci_low``, and ``ci_high`` are ``rebalancing_report.summary``'s
    mean over trials and its interval. Against ``REFERENCE_ARM`` each other
    simulator gets ``rebalancing_report.comparisons``' median paired
    difference (``difference``, ``difference_ci_low``, ``difference_ci_high``,
    ``p_value``) and ``trials_better``, the trials in which it did better in
    the reading's better direction of ``METRIC_LABELS``; a tie is not better.
    The reference arm's difference columns are empty.
    """
    drawn = _of_arms(per_trial, styles)
    levels = summary(drawn)
    differences = comparisons(drawn).rename(
        columns={
            "median_difference": "difference",
            "ci_low": "difference_ci_low",
            "ci_high": "difference_ci_high",
        }
    )
    better = []
    for metric in metrics:
        wide = drawn[metric].unstack("arm")
        change = wide.drop(columns=REFERENCE_ARM).sub(wide[REFERENCE_ARM], axis=0)
        wins = change > 0 if METRIC_LABELS[metric][2] == "higher" else change < 0
        better.append(
            wins.sum().rename("trials_better").reset_index().assign(metric=metric)
        )
    table = levels[levels["metric"].isin(metrics)].merge(
        differences[
            [
                "metric",
                "arm",
                "difference",
                "difference_ci_low",
                "difference_ci_high",
                "p_value",
            ]
        ],
        on=["metric", "arm"],
        how="left",
    )
    return table.merge(
        pd.concat(better, ignore_index=True), on=["metric", "arm"], how="left"
    )


def size_change_frame(runs: pd.DataFrame, sizes: Sequence[int]) -> pd.DataFrame:
    """Per trial and arm, every reading's change from the smallest to the largest size.

    The change is the least-squares slope of the reading over ``sizes`` times
    their span, so it rests on every size rather than on the two ends alone,
    whose trading-window moves would carry over into it unaveraged. Indexed
    and labeled like ``per_trial_frame``, so ``performance_table`` reads it as
    it reads levels.

    Raises
    ------
    ValueError
        If a trial lacks an arm at one of ``sizes``, which would bias its slope.
    """
    readings = pd.concat(
        {size: per_trial_frame(runs[runs["size"] == size]) for size in sizes},
        names=["size"],
    )
    counts = readings.groupby(["trial", "arm"]).size()
    if (counts != len(sizes)).any():
        raise ValueError(
            f"Every trial and arm needs a run at each of the sizes {list(sizes)}; "
            f"{list(counts[counts != len(sizes)].index)} do not have one."
        )
    centered = np.asarray(sizes, dtype=float) - np.mean(sizes)
    weights = pd.Series(
        centered / (centered @ centered) * (max(sizes) - min(sizes)),
        index=pd.Index(sizes, name="size"),
    )
    changes = (
        readings[list(METRICS)]
        .mul(weights, axis=0, level="size")
        .groupby(["trial", "arm"])
        .sum()
    )
    return changes.join(
        readings.groupby(["trial", "arm"])[["fit_type", "device"]].first()
    )


def universe_size_table(
    runs: pd.DataFrame, styles: Sequence[BlogStyle], sizes: Sequence[int]
) -> pd.DataFrame:
    """``performance_table`` of ``UNIVERSE_SIZE_METRICS`` at every size and of their change.

    ``runs`` holds the same trials at every size (``complete_universe_runs``),
    so a simulator's curve over the sizes changes only with the assets traded.
    Rows with ``reading`` ``SIZE_READING`` hold the levels at ``size``. Rows
    with ``CHANGE_READING`` and no size read ``size_change_frame`` instead:
    ``mean`` and its interval are a simulator's mean change over the sizes,
    ``difference`` the median over trials of how much more its reading changed
    than ``REFERENCE_ARM``'s, with interval and p-value, and
    ``trials_better`` the trials in which its change beat the reference's in
    the reading's better direction.
    """
    at_sizes = [
        performance_table(
            per_trial_frame(runs[runs["size"] == size]),
            styles,
            UNIVERSE_SIZE_METRICS,
        ).assign(reading=SIZE_READING, size=size)
        for size in sizes
    ]
    change = performance_table(
        size_change_frame(runs, sizes), styles, UNIVERSE_SIZE_METRICS
    ).assign(reading=CHANGE_READING, size=pd.NA)
    return pd.concat([*at_sizes, change], ignore_index=True).astype({"size": "Int64"})


def check_rates(rates_path: Path, identity: dict) -> None:
    """Refuse a risk-free rate file other than the one the backtests' cash earned.

    Raises
    ------
    ValueError
        If the SHA-256 of ``rates_path`` differs from the rebalancing results'
        identity.
    """
    recorded = identity["risk_free_rates"]
    if hashlib.sha256(rates_path.read_bytes()).hexdigest() != recorded["sha256"]:
        raise ValueError(
            f"{rates_path} is not the risk-free rate file {recorded['file']} the "
            f"backtests' cash earned (its SHA-256 differs), so the risk-free "
            f"return cannot be recomputed from it."
        )


def risk_free_return(
    runs: pd.DataFrame, dates: pd.DatetimeIndex, rates: pd.Series
) -> float:
    """Cash's annualized return over each run's trading window, averaged over the trials.

    Cash earns ``risk_free.cash_returns`` on every trading day of ``dates`` after
    the window's first, the days a run's daily returns cover, and compounds to
    an annual rate over those days exactly as ``annualized_return`` does, so the
    two read on one scale. Every arm of a trial trades the same window.

    Raises
    ------
    ValueError
        If the arms of one trial traded different windows, or as
        ``risk_free.cash_returns`` raises for a day without a recent rate.
    """
    columns = ["trading_start", "trading_end"]
    if (runs.groupby("trial")[columns].nunique() > 1).any(axis=None):
        raise ValueError("The arms of one trial traded different windows.")
    annual = []
    for start, end in runs.groupby("trial")[columns].first().itertuples(index=False):
        days = dates[(dates > start) & (dates <= end)]
        growth = np.prod(1 + risk_free.cash_returns(rates, days, "LINEAR"))
        annual.append(growth ** (risk_free.TRADING_DAYS_PER_YEAR / len(days)) - 1)
    return float(np.mean(annual))


def risk_return_table(
    performance: pd.DataFrame, styles: Sequence[BlogStyle], risk_free_annual: float
) -> pd.DataFrame:
    """Per drawn simulator: mean annualized volatility and return, and the frontier with cash.

    The volatility and return are ``performance_table``'s means over trials.
    Mixing a portfolio with cash, which earns ``risk_free_annual`` without
    volatility, reaches every point of the line from cash through the
    portfolio's point; ``excess_per_volatility`` is that line's slope, the mean
    return above ``risk_free_annual`` per unit of mean volatility. ``frontier``
    marks the simulator with the steepest line: no mix of cash with another
    simulator's portfolio rises above it. A last row, ``RISK_FREE_ROW``, holds
    cash itself.
    """
    means = performance.set_index(["metric", "arm"])["mean"]
    table = pd.DataFrame(
        {
            "arm": [style.name for style in styles],
            "annualized_volatility": [
                means["annualized_volatility", style.name] for style in styles
            ],
            "annualized_return": [
                means["annualized_return", style.name] for style in styles
            ],
        }
    )
    table["excess_per_volatility"] = (
        table["annualized_return"] - risk_free_annual
    ) / table["annualized_volatility"]
    table["frontier"] = table.index == table["excess_per_volatility"].idxmax()
    cash = pd.DataFrame(
        [
            {
                "arm": RISK_FREE_ROW,
                "annualized_volatility": 0.0,
                "annualized_return": risk_free_annual,
                "frontier": False,
            }
        ]
    )
    table = pd.concat([table, cash], ignore_index=True)
    table["trials"] = int(performance["trials"].iloc[0])
    return table


def _legend(
    figure, styles: Sequence[BlogStyle], *, lines: bool, reference: str | None
) -> None:
    """One entry per simulator below the panels, as a line or as a marker.

    A last entry names the dark horizontal line every panel draws
    (``_target_line``) as ``reference``; a label inside the panels would
    collide with the marks near it. ``None`` is for a figure without that line.
    """
    handles = [
        Line2D(
            [],
            [],
            color=style.color,
            linewidth=LINE_WIDTH if lines else 0,
            marker="o",
            markersize=8,
            markerfacecolor=style.color,
            markeredgecolor=SURFACE,
            markeredgewidth=MARKER_EDGE,
            label=style.inline_label,
        )
        for style in styles
    ]
    if reference is not None:
        handles.append(
            Line2D([], [], color=INK_SECONDARY, linewidth=1, label=reference)
        )
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        frameon=False,
        fontsize=TEXT_SIZE,
        labelcolor=INK_SECONDARY,
    )


def _end_labels(ax, labels: Sequence[tuple[float, float, str]]) -> None:
    """Label line ends at (x, y, text), pushed apart so no two labels overlap.

    Call it once the axis limits are final: the gap between labels is
    ``LABEL_GAP`` of the axis height, and the labels keep their order by y.
    """
    low, high = ax.get_ylim()
    gap = LABEL_GAP * (high - low)
    ordered = sorted(labels, key=itemgetter(1))
    placed: list[float] = []
    for _, y, _ in ordered:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    # An annotation above the axis is not drawn, so the stack moves down instead.
    overflow = max(placed[-1] - high, 0.0)
    for (x, _, text), y in zip(ordered, placed):
        ax.annotate(
            text,
            xy=(x, y - overflow),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=TEXT_SIZE,
        )


def plot_ranking(
    ranks: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Each simulator's share of trials won per score, best mean rank on top."""
    by_name = {style.name: style for style in styles}
    figure, axes = _figure(12, 0.6 * len(styles) + 2.4, cols=len(BLOG_RANKING_SCORES))
    for ax, (score, (title, reads)) in zip(axes.flat, BLOG_RANKING_SCORES.items()):
        rows = ranks[ranks["score"] == score].set_index("arm")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        for position, arm in enumerate(rows.index):
            share = rows.loc[arm, "share_won"]
            ax.barh(position, share, height=0.55, color=by_name[arm].color)
            ax.text(
                share + 0.02,
                position,
                f"{PERCENT * share:.0f} % · mean rank {rows.loc[arm, 'mean_rank']:.2f}",
                va="center",
                color=INK_SECONDARY,
                fontsize=TEXT_SIZE,
            )
        ax.set_yticks(
            range(len(rows)), [by_name[arm].inline_label for arm in rows.index]
        )
        ax.invert_yaxis()
        # Room right of a full bar for its label; ticks stop at 100 %.
        ax.set_xlim(0, 1.55)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        _panel_title(
            ax,
            f"{title}: {reads}",
            f"share of {int(rows['trials'].iloc[0])} trials with the best "
            f"next-day score",
        )
    _headline(figure, "Most realistic next-day scenarios per trial")
    _style_text(figure)
    return _save(figure, directory, "ranking")


def plot_skill_horizons(
    skills: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Median skill against replaying the window over the horizons, one panel per score."""
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    reference = next(style for style in styles if style.name == REFERENCE_ARM)
    horizons = sorted(skills["horizon"].unique())
    figure, axes = _figure(12, 5.2, cols=len(BLOG_SKILL_SCORES))
    for ax, (score, (title, reads)) in zip(axes.flat, BLOG_SKILL_SCORES.items()):
        table = skills[skills["score"] == score].rename(
            columns={"median_skill": "estimate"}
        )
        _curves(ax, table, "horizon", contenders)
        _target_line(ax, 0)
        ax.set_xlim(right=horizons[-1] * END_LABEL_ROOM)
        last = table[table["horizon"] == horizons[-1]].set_index("arm")
        _end_labels(
            ax,
            [
                (
                    horizons[-1] * 1.1,
                    last.loc[style.name, "estimate"],
                    style.inline_label,
                )
                for style in contenders
            ],
        )
        _axis_labels(
            ax,
            "forecast horizon (trading days)",
            f"% better than {reference.inline_label}",
        )
        _panel_title(
            ax,
            f"{title}: {reads}",
            "median over trials with its 95 % bootstrap interval",
        )
    _headline(figure, "Score improvement over replaying the window, by horizon")
    _legend(
        figure,
        contenders,
        lines=True,
        reference=f"{reference.inline_label}, replaying the window",
    )
    _style_text(figure)
    return _save(figure, directory, "skill_horizons")


def plot_growth(
    growth: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Mean cumulative return and mean drawdown after every step, one line per simulator.

    The drawdown panel averages each trial's own drawdown, not the drawdown of
    the mean return curve, so it deepens while the mean return rises.
    """
    figure, axes = _figure(10, 7.5, rows=2)
    top, bottom = axes[:, 0]
    bottom.sharex(top)
    labels = []
    for style in styles:
        rows = growth[growth["arm"] == style.name]
        days = rows["trading_day"].to_numpy()
        returns = PERCENT * rows["cumulative_return"].to_numpy()
        for ax, values in (
            (top, returns),
            (bottom, -PERCENT * rows["drawdown"].to_numpy()),
        ):
            ax.plot(
                days,
                values,
                color=style.color,
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                zorder=3,
            )
        labels.append(
            (days[-1], returns[-1], f"{style.inline_label} {returns[-1]:+.1f} %")
        )
    steps = growth["trading_day"].unique()
    top.set_xticks(steps[::GROWTH_TICK_STEPS])
    top.set_xlim(steps[0], steps[-1] * 1.25)
    top.tick_params(labelbottom=False)
    _end_labels(top, labels)
    trials = int(growth["trials"].iloc[0])
    _axis_labels(top, "", "cumulative return (%)")
    _panel_title(
        top, "Cumulative return", f"mean over {trials} trials at every rebalancing date"
    )
    _axis_labels(bottom, "trading days since the backtest started", "drawdown (%)")
    _panel_title(
        bottom,
        "Drawdown of each trial from its own peak",
        f"mean over {trials} trials of how far each lies below its highest value so far",
    )
    _headline(figure, "Rebalanced portfolio over the backtest")
    _legend(figure, styles, lines=True, reference=None)
    _style_text(figure)
    return _save(figure, directory, "growth")


def _level(performance: pd.DataFrame, metric: str, arm: str) -> np.ndarray:
    """(mean, interval low, interval high) of a reading, in display units."""
    row = performance[(performance["metric"] == metric) & (performance["arm"] == arm)]
    scale = METRIC_LABELS[metric][3]
    return scale * row[["mean", "ci_low", "ci_high"]].to_numpy(dtype=float)[0]


def _dot_groups(
    points: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]
) -> list[list[str]]:
    """Rows of ``points`` (x, y per row) grouped where their dots would hide each other.

    A row joins the first group whose first row lies within ``DOT_OVERLAP`` of
    the axis span on both axes, or else starts a group; groups and their
    members keep the rows' order.
    """
    spans = DOT_OVERLAP * np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    groups: list[list[str]] = []
    for row in points.index:
        for group in groups:
            if (abs(points.loc[row] - points.loc[group[0]]).to_numpy() < spans).all():
                group.append(row)
                break
        else:
            groups.append([row])
    return groups


def plot_risk_return(
    table: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Each simulator and cash as a dot at mean annualized volatility and return, with the frontier.

    The frontier is the line from cash through the simulator ``frontier`` marks
    in ``risk_return_table``: solid where a mix of its portfolio with cash
    reaches it, dashed beyond, where the mix borrows at the risk-free rate. The
    area below the line is shaded. Dots that would hide each other
    (``_dot_groups``) draw as concentric rings, the first simulator outermost,
    under one label listing them outside in. The two dots on the line keep
    their labels off it: cash's below, the frontier simulator's to its left.
    """
    figure, axes = _figure(8, 6.5)
    ax = axes[0, 0]
    ax.grid(axis="x", color=GRID, linewidth=1)
    points = (
        PERCENT * table.set_index("arm")[["annualized_volatility", "annualized_return"]]
    )
    cash = points.loc[RISK_FREE_ROW, "annualized_return"]
    best = next(
        style
        for style in styles
        if table.loc[table["arm"] == style.name, "frontier"].item()
    )
    best_volatility, best_return = points.loc[best.name]
    slope = (best_return - cash) / best_volatility
    right = (1 + FRONTIER_ROOM) * points["annualized_volatility"].max()
    edge = cash + slope * right
    drawn = [*points["annualized_return"], edge]
    room = FRONTIER_ROOM * (max(drawn) - min(drawn))
    bottom, top = min(drawn) - room, max(drawn) + room
    ax.fill_between(
        [0, right],
        [cash, edge],
        bottom,
        color=best.color,
        alpha=FRONTIER_SHADE,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        [0, best_volatility],
        [cash, best_return],
        color=best.color,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    ax.plot(
        [best_volatility, right],
        [best_return, edge],
        color=best.color,
        linewidth=LINE_WIDTH,
        linestyle="--",
        zorder=2,
    )
    ax.annotate(
        f"cash mixed with {best.inline_label}",
        xy=(right, edge),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=best.color,
        fontsize=TEXT_SIZE,
    )
    ax.set_xlim(-FRONTIER_ROOM * right / 4, right)
    ax.set_ylim(bottom, top)
    colors = {style.name: style.color for style in styles}
    colors[RISK_FREE_ROW] = INK_SECONDARY
    labels = {style.name: style.inline_label for style in styles}
    labels[RISK_FREE_ROW] = f"cash, risk-free {cash:.1f} %"
    right_x, up_y = DOT_LABEL_OFFSET
    placements = {
        RISK_FREE_ROW: ((right_x, -up_y), "left", "top"),
        best.name: ((-right_x, up_y), "right", "bottom"),
    }
    for group in _dot_groups(points.loc[list(colors)], ax.get_xlim(), ax.get_ylim()):
        for rank, arm in enumerate(group):
            volatility, annual = points.loc[arm]
            ax.scatter(
                [volatility],
                [annual],
                s=INTERVAL_MARKER_AREA * (len(group) - rank) ** 2 / len(group),
                facecolor=colors[arm],
                edgecolor=SURFACE,
                linewidth=MARKER_EDGE,
                zorder=3,
            )
        offset, ha, va = placements.get(group[0], (DOT_LABEL_OFFSET, "left", "bottom"))
        ax.annotate(
            "\n".join(labels[arm] for arm in group),
            xy=tuple(points.loc[group[0]]),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            color=INK_SECONDARY,
            fontsize=TEXT_SIZE,
        )
    trials = int(table["trials"].iloc[0])
    _axis_labels(ax, "annualized volatility (%)", "annualized return (%)")
    _panel_title(
        ax,
        "Return against risk, with cash",
        f"mean over {trials} trials; up and left is better",
    )
    _headline(figure, "Rebalanced portfolios: return against volatility, with cash")
    _style_text(figure)
    return _save(figure, directory, "risk_return")


def plot_performance(
    performance: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """One panel per reading of ``PERFORMANCE_METRICS``: each simulator's mean and interval.

    A vertical line marks ``REFERENCE_ARM``'s mean, and each other simulator's
    label counts the trials in which it did better than the reference.
    """
    cols = 3
    rows = -(-len(PERFORMANCE_METRICS) // cols)
    figure, axes = _figure(15, rows * (0.75 * len(styles) + 2.2), rows=rows, cols=cols)
    reference = next(style for style in styles if style.name == REFERENCE_ARM)
    for ax, metric in zip(axes.flat, PERFORMANCE_METRICS):
        title, _, better, scale = METRIC_LABELS[metric]
        levels = [_level(performance, metric, style.name) for style in styles]
        _interval_panel(
            ax,
            [(style, *level) for style, level in zip(styles, levels)],
            reference=_level(performance, metric, REFERENCE_ARM)[0],
            reference_label=reference.inline_label,
        )
        counts = performance[performance["metric"] == metric].set_index("arm")
        ax.set_yticks(
            range(len(styles)),
            [
                style.inline_label
                if style.name == REFERENCE_ARM
                else f"{style.inline_label}\nbetter in "
                f"{int(counts.loc[style.name, 'trials_better'])} of "
                f"{int(counts.loc[style.name, 'trials'])}"
                for style in styles
            ],
        )
        unit, decimals = (
            (" %", PERCENT_DECIMALS) if scale == PERCENT else ("", RATIO_DECIMALS)
        )
        low = min(level[1] for level in levels)
        high = max(level[2] for level in levels)
        ax.set_xlim(low - 0.05 * (high - low), high + VALUE_LABEL_ROOM * (high - low))
        for position, level in enumerate(levels):
            ax.annotate(
                f"{level[0]:.{decimals}f}{unit}",
                xy=(level[2], position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=TEXT_SIZE,
            )
        ax.set_xlabel(LEVEL_UNITS[metric], color=INK_SECONDARY, fontsize=TEXT_SIZE)
        _panel_title(ax, title, f"{better} is better")
    for ax in list(axes.flat)[len(PERFORMANCE_METRICS) :]:
        ax.set_visible(False)
    _headline(figure, "Performance and risk of the rebalanced portfolios")
    _style_text(figure)
    return _save(figure, directory, "performance")


def plot_universe_sizes(
    table: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Mean total return over the universe sizes, ``FOCUS_FIT_TYPE`` against the others.

    Only the means are drawn: the trading windows' market moves spread every
    simulator's level alike, so intervals of the levels overlap even where the
    paired differences of the CSV are clear. The other simulators are drawn
    thin over the band between their lowest and highest mean, so they read as
    one group against the simulator the post is about.

    Raises
    ------
    ValueError
        If ``styles`` holds no ``FOCUS_FIT_TYPE`` simulator.
    """
    focus = [style for style in styles if style.fit_type == FOCUS_FIT_TYPE]
    if len(focus) != 1:
        raise ValueError(
            f"The universe-size figure draws one {FOCUS_FIT_TYPE} simulator "
            f"against the others; the results hold {len(focus)}."
        )
    group = [style for style in styles if style.fit_type != FOCUS_FIT_TYPE]
    metric = UNIVERSE_SIZE_METRICS[0]
    levels = table[(table["reading"] == SIZE_READING) & (table["metric"] == metric)]
    means = PERCENT * levels.pivot(index="size", columns="arm", values="mean")
    means = means.sort_index()
    sizes = means.index.to_numpy(dtype=float)
    figure, axes = _figure(10, 5.5)
    ax = axes[0, 0]
    names = [style.name for style in group]
    ax.fill_between(
        sizes,
        means[names].min(axis=1),
        means[names].max(axis=1),
        color=MUTED,
        alpha=GROUP_SHADE,
        linewidth=0,
    )
    for style in group:
        ax.plot(sizes, means[style.name], color=style.color, linewidth=GROUP_LINE_WIDTH)
    ax.plot(
        sizes,
        means[focus[0].name],
        color=focus[0].color,
        linewidth=FOCUS_LINE_WIDTH,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        zorder=3,
    )
    ax.set_xticks(sizes, [f"{size:.0f}" for size in sizes])
    ax.set_xlim(sizes[0], sizes[-1] + (sizes[-1] - sizes[0]) * 0.3)
    last = means.iloc[-1]
    _end_labels(
        ax,
        [
            (
                sizes[-1],
                last[style.name],
                f"{style.inline_label} {last[style.name]:+.1f} %",
            )
            for style in styles
        ],
    )
    trials = int(levels["trials"].iloc[0])
    _axis_labels(ax, "assets in the universe", "total return (%)")
    _panel_title(
        ax,
        "Total return over the backtest",
        f"mean over {trials} trials, each trading nested subsets of its assets "
        f"over one window",
    )
    _headline(figure, "Rebalanced portfolio by number of assets")
    _style_text(figure)
    return _save(figure, directory, "universe_sizes")


def write_blog(config: ExperimentConfig, devices_results: Path) -> ReportOutputs:
    """Compute the blog's readings from a configuration's results and write its figures.

    The scenario figures read ``config.output`` and the price file it was
    measured on, over the trials in which every arm has every origin at every
    horizon; the rebalancing figures read ``config.rebalancing.output``, over
    the trials in which every arm finished, and the risk-free rate file the
    backtests' cash earned; the universe-size figure reads
    ``config.universe_sizes.output``, over the trials in which every arm
    finished at every size. The speed figures read ``devices_results``, the
    results of a CPU-against-GPU configuration (``devices_report``).
    Everything goes to ``config.blog_dir``.

    Raises
    ------
    FileNotFoundError
        If ``devices_results`` does not exist.
    ValueError
        If the scenario results hold no scenario or portfolio scores, either
        results file was measured on another price file, the rebalancing
        results earned cash at another risk-free rate file, the scenario
        results lack ``REFERENCE_ARM`` or have too few complete trials, the
        rebalancing results have too few complete trials, the universe-size
        results are missing, were measured on another price or rate file, or
        have too few trials complete at every size, or ``devices_results``
        lacks a session or a trial both sessions finished.
    """
    frame, metadata = read_results(config.output)
    identity = metadata.identity
    _check_scores_recorded(identity)
    check_prices(config.prices, identity)
    experiment = identity["experiment"]
    frame, excluded = complete_trials(
        frame,
        tuple(identity["arms"]),
        experiment["origins"],
        len(experiment["horizons"]),
    )
    per_trial = {
        horizon: aggregate(frame[frame["horizon"] == horizon])
        for horizon in experiment["horizons"]
    }
    styles = blog_styles(per_trial[NEXT_DAY])
    prices = load_prices(config.prices)

    runs, run_metadata = read_results(config.rebalancing.output)
    check_prices(config.prices, run_metadata.identity)
    check_rates(config.rebalancing.risk_free_rates, run_metadata.identity)
    runs = complete_runs(runs, tuple(run_metadata.identity["arms"]))
    run_styles = blog_styles(per_trial_frame(runs))
    step_days = run_metadata.identity["rebalancing"]["look_forward_window"]
    performance = performance_table(
        per_trial_frame(runs), run_styles, PERFORMANCE_METRICS
    )
    sweep = config.universe_sizes
    sized_runs, sized_metadata = read_results(sweep.output)
    check_prices(config.prices, sized_metadata.identity)
    check_rates(config.rebalancing.risk_free_rates, sized_metadata.identity)
    sized_runs = complete_universe_runs(
        sized_runs, tuple(sized_metadata.identity["arms"]), sweep.sizes
    )
    sized_styles = blog_styles(per_trial_frame(sized_runs))
    cash = risk_free_return(
        runs,
        prices.index,
        risk_free.load_risk_free_rates(config.rebalancing.risk_free_rates),
    )

    tables = {
        "ranking": ranking_table(per_trial[NEXT_DAY], styles),
        "skill_horizons": skill_table(per_trial, styles),
        "growth": growth_table(runs, run_styles, step_days),
        "risk_return": risk_return_table(performance, run_styles, cash),
        "performance": performance,
        "universe_sizes": universe_size_table(sized_runs, sized_styles, sweep.sizes),
    }
    speed = device_tables(devices_results)
    directory = config.blog_dir
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    files += plot_ranking(tables["ranking"], styles, directory)
    files += plot_skill_horizons(tables["skill_horizons"], styles, directory)
    files += plot_growth(tables["growth"], run_styles, directory)
    files += plot_risk_return(tables["risk_return"], run_styles, directory)
    files += plot_performance(tables["performance"], run_styles, directory)
    files += plot_universe_sizes(tables["universe_sizes"], sized_styles, directory)
    files += write_device_figures(speed, directory)
    return ReportOutputs(files=files, excluded_trials=excluded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.blog",
        description="Blog figures of a configuration's scenario and rebalancing "
        "results, written to <output stem>_blog beside its results; runs nothing.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    parser.add_argument(
        "--devices-config",
        type=Path,
        required=True,
        help="YAML configuration of the CPU-against-GPU experiment "
        "(task experiments:devices), whose results the speed figures read.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    devices, _ = load_devices_config(args.devices_config)
    outputs = write_blog(config, devices.output)
    if outputs.excluded_trials:
        print(
            f"Left out {len(outputs.excluded_trials)} incomplete scenario trials: "
            f"{outputs.excluded_trials}"
        )
    print(f"Blog figures finished: {len(outputs.files)} files in {config.blog_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
