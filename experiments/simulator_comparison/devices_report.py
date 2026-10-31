# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables and figures of the CPU-against-GPU experiment (``devices.py``).

``device_tables`` reads a results file into three tables, each holding exactly
the numbers its figure draws:

- ``runtime``: each simulator's mean seconds per scenario-generation call on
  every device it ran on.
- ``runtime_devices``: a rebalancing decision's stages per device (scenario
  generation, optimizer, and their total) and the CPU-to-GPU ratio of each.
- ``solvers``: each paired solver's mean seconds per solve of the decisions'
  identical problems, its time over the GPU pipeline's solver, and how far its
  solutions lie from that solver's.

A mean is the mean over trials of each trial's mean, with its 95 % bootstrap
interval over trials (``analysis.bootstrap_interval``); a ratio is the median
over trials of each trial's ratio of means, with its interval. Only trials
that both sessions finished are read, and warm-up rows are left out of every
timing. Every table names the hardware: the host's CPU model, sockets, and
physical cores, and the GPU the GPU session ran on.

``write_device_figures`` writes the tables as CSV files and draws them in the
blog's style (``blog_style``). The blog writes them beside its other figures,
and this module's ``main`` writes them as the experiment's own report.
Titles state what is drawn, never a finding.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import BOOTSTRAP_SEED, bootstrap_interval
from .arms import ARMS_BY_NAME
from .blog_style import (
    BLOG_NAMES,
    DECADE_TICK_STEPS,
    SECONDS_TICK_STEPS,
    TEXT_SIZE,
    _footnote,
    _headline,
    _panel_title,
    _seconds_rows,
    _style_text,
)
from .config import load_devices_config
from .devices import DEVICES
from .plotting import (
    AXIS,
    DEVICE_ORDER,
    INK,
    INK_SECONDARY,
    LINE_WIDTH,
    MARKER_EDGE,
    SIMULATOR_COLORS,
    SIMULATOR_ORDER,
    SURFACE,
    _figure,
    _save,
)
from .report import _duration
from .solvers import SOLVERS_BY_NAME
from .table import read_results

# Each stage column of a decision row, as the figures name it.
STAGES = {
    "generation_seconds": "Scenarios",
    "optimizer_seconds": "Optimizer",
    "seconds": "Decision",
}
# The stages a decision's bar stacks, bottom to top; they sum to "seconds".
STACKED_STAGES = ("generation_seconds", "optimizer_seconds")
# Bar width and room right of the last bar for its labels, in bar positions.
STACK_BAR_WIDTH = 0.5
STACK_RIGHT_ROOM = 1.3
STACK_LABEL_OFFSET = 0.08
# Room above the slowest total's interval, and the least distance between two
# labels of one bar, as factors of the axis height.
STACK_TOP_ROOM = 1.06
STACK_LABEL_SPACING = 0.13
# Inches per row of the runtime and solvers figures.
SECONDS_ROW_HEIGHT = 0.4
# The reading of runtime_devices.csv that holds the CPU-to-GPU time ratio.
DEVICE_RATIO = "cpu_over_gpu"
# Slots of the reference palette (plotting.SIMULATOR_COLORS): JointFM's blue
# for its scenarios, the swap cells' amber for the optimizer, ink for the total.
STAGE_COLORS = {
    "generation_seconds": "#2a78d6",
    "optimizer_seconds": "#eda100",
    "seconds": INK,
}
# One reference-palette slot per solver, its cuOpt methods in the blue-violet
# range and the CPU solvers in the warm and green slots.
SOLVER_COLORS = {
    "CUOPT": "#2a78d6",
    "CUOPT_BARRIER": "#6250d6",
    "CUOPT_DUAL_SIMPLEX": "#e87ba4",
    "HIGHS": "#eb6834",
    "CLARABEL": "#1baf7a",
}
TABLES = ("runtime", "runtime_devices", "solvers")


def read_device_results(path: Path) -> tuple[pd.DataFrame, dict]:
    """The rows of the trials both sessions finished, and the file's identity.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If either session is missing, or no trial is finished by both.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run task experiments:devices with a GPU and "
            f"a CPU service first."
        )
    frame, metadata = read_results(path)
    identity = metadata.identity
    sessions = {arm["device"] for arm in identity["arms"].values()}
    missing = [device for device in DEVICES if device not in sessions]
    if missing:
        raise ValueError(
            f"{path} holds no {missing} session; task experiments:devices runs both."
        )
    decisions = frame[frame["kind"] == "decision"]
    finished = set.intersection(
        *(set(decisions.loc[decisions["session"] == d, "trial"]) for d in DEVICES)
    )
    if not finished:
        raise ValueError(f"{path} holds no trial that both sessions finished.")
    return frame[frame["trial"].isin(finished)], identity


def hardware(identity: dict) -> dict[str, object]:
    """The host's CPU and the GPU session's GPU, as every table records them."""
    host = identity["host"]
    gpu = next(
        record["gpu"] for record in identity["services"].values() if "gpu" in record
    )
    return {
        "cpu": host["cpu"],
        "cpu_sockets": host["sockets"],
        "cpu_cores": host["physical_cores"],
        "gpu": gpu["name"],
    }


def _hardware_text(table: pd.DataFrame) -> str:
    first = table.iloc[0]
    sockets = int(first["cpu_sockets"])
    return (
        f"CPU: {f'{sockets} × ' if sockets > 1 else ''}{first['cpu']}, "
        f"{int(first['cpu_cores'])} cores in all. GPU: one {first['gpu']}."
    )


def _mean(values: pd.Series) -> dict:
    """Mean of per-trial values, with its bootstrap interval over trials."""
    array = values.dropna().to_numpy()
    low, high = bootstrap_interval(
        array, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "estimate": float(array.mean()),
        "ci_low": low,
        "ci_high": high,
        "trials": len(array),
    }


def _ratio(numerator: pd.Series, denominator: pd.Series) -> dict:
    """Median over trials of per-trial ratios, with its bootstrap interval."""
    ratios = (numerator / denominator).dropna().to_numpy()
    low, high = bootstrap_interval(
        ratios, np.median, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "estimate": float(np.median(ratios)),
        "ci_low": low,
        "ci_high": high,
        "trials": len(ratios),
    }


def _timed(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    return frame[(frame["kind"] == kind) & ~frame["warmup"].astype(bool)]


def runtime_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each simulator's mean seconds per generation call, per device, over the decisions.

    ``num_scen`` is the median scenario count a call returned: the configured
    count, except for Historical, which replays the look-back window's days.
    """
    calls = _timed(frame, "generation")
    per_trial = calls.groupby(["name", "trial"])["seconds"].mean()

    def order(name: str) -> tuple[int, int]:
        arm = ARMS_BY_NAME[name]
        return SIMULATOR_ORDER.index(arm.fit_type), DEVICE_ORDER.index(arm.device)

    rows = [
        {
            "arm": name,
            "fit_type": ARMS_BY_NAME[name].fit_type,
            "device": ARMS_BY_NAME[name].device,
            **_mean(per_trial[name]),
            "num_scen": int(calls.loc[calls["name"] == name, "num_scen"].median()),
        }
        for name in sorted(calls["name"].unique(), key=order)
    ]
    return pd.DataFrame(rows).assign(
        assets=identity["experiment"]["assets"], **hardware(identity)
    )


def stage_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each stage of a pipeline decision per device, and its CPU-to-GPU ratio.

    Rows of reading ``seconds`` hold a device's mean seconds per decision of
    the stage; the row of reading ``DEVICE_RATIO`` holds the median over
    trials of the CPU's mean over the GPU's.
    """
    decisions = _timed(frame, "decision")
    per_trial = decisions.groupby(["session", "trial"])[list(STAGES)].mean()
    sessions = {arm["device"]: (name, arm) for name, arm in identity["arms"].items()}
    rows = []
    for stage in STAGES:
        for device in DEVICE_ORDER:
            name, arm = sessions[device]
            rows.append(
                {
                    "stage": stage,
                    "reading": "seconds",
                    "device": device,
                    "arm": name,
                    "solver": arm["solver"],
                    **_mean(per_trial.loc[device, stage]),
                }
            )
        rows.append(
            {
                "stage": stage,
                "reading": DEVICE_RATIO,
                "device": "",
                "arm": "",
                "solver": "",
                **_ratio(per_trial.loc["CPU", stage], per_trial.loc["GPU", stage]),
            }
        )
    return pd.DataFrame(rows).assign(
        num_scen=identity["devices"]["num_scen"],
        assets=identity["experiment"]["assets"],
        **hardware(identity),
    )


def solver_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each paired solver's seconds per solve, its time over the GPU solver's, and its distance.

    ``estimate`` is the mean seconds per solve and ``setup_seconds`` the mean
    seconds of the problem's setup; ``ratio`` is the median over trials of
    the solver's mean over the GPU pipeline solver's. ``max_weight_difference``
    is the largest absolute difference of an asset or cash weight from the GPU
    solver's solution of the same problem, and ``max_objective_gap`` the
    largest absolute difference of the objective, both over every decision.
    """
    solves = frame[frame["kind"] == "solve"]
    timed = solves[~solves["warmup"].astype(bool)]
    per_trial = timed.groupby(["name", "trial"])[["seconds", "setup_seconds"]].mean()
    reference_name = identity["devices"]["gpu_solver"]
    keys = ["trial", "decision"]
    reference = solves[solves["name"] == reference_name].set_index(keys)
    rows = []
    for name in identity["devices"]["paired_solvers"]:
        joined = (
            solves[solves["name"] == name]
            .set_index(keys)
            .join(reference, rsuffix="_reference", how="inner")
        )
        weights = np.abs(
            np.stack(joined["weights"].to_numpy())
            - np.stack(joined["weights_reference"].to_numpy())
        ).max(axis=1)
        cash = np.abs(joined["cash"] - joined["cash_reference"]).to_numpy()
        ratio = _ratio(
            per_trial.loc[name, "seconds"], per_trial.loc[reference_name, "seconds"]
        )
        rows.append(
            {
                "solver": name,
                "label": SOLVERS_BY_NAME[name].label,
                "device": SOLVERS_BY_NAME[name].device,
                **_mean(per_trial.loc[name, "seconds"]),
                "setup_seconds": float(per_trial.loc[name, "setup_seconds"].mean()),
                "ratio": ratio["estimate"],
                "ratio_ci_low": ratio["ci_low"],
                "ratio_ci_high": ratio["ci_high"],
                "max_weight_difference": float(np.maximum(weights, cash).max()),
                "max_objective_gap": float(
                    np.abs(joined["objective"] - joined["objective_reference"]).max()
                ),
                "decisions": len(joined),
            }
        )
    return pd.DataFrame(rows).assign(
        reference=reference_name,
        num_scen=identity["devices"]["num_scen"],
        assets=identity["experiment"]["assets"],
        **hardware(identity),
    )


def device_tables(results: Path) -> dict[str, pd.DataFrame]:
    """Every table of the module docstring, by name, from a results file.

    Raises
    ------
    FileNotFoundError, ValueError
        As ``read_device_results`` raises.
    """
    frame, identity = read_device_results(results)
    return {
        "runtime": runtime_table(frame, identity),
        "runtime_devices": stage_table(frame, identity),
        "solvers": solver_table(frame, identity),
    }


def plot_runtime(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Each simulator's mean seconds per scenario-generation call per device (``runtime_table``).

    The calls span several decades, so only whole decades are labeled.
    """
    figure, axes = _figure(8, SECONDS_ROW_HEIGHT * len(table) + 2.8)
    ax = axes[0, 0]
    _seconds_rows(
        ax,
        [
            (
                f"{BLOG_NAMES[row.fit_type]} on {row.device}",
                SIMULATOR_COLORS[row.fit_type],
                row.estimate,
                row.ci_low,
                row.ci_high,
            )
            for row in table.itertuples()
        ],
        "call",
        DECADE_TICK_STEPS,
    )
    jointfm = table[table["fit_type"] == "jointfm"].iloc[0]
    _panel_title(
        ax,
        "Seconds per scenario-generation call",
        f"mean over {int(table['trials'].min())} trials with its 95 % interval;"
        " calls without warm-up",
    )
    _footnote(
        figure,
        f"A call draws {int(jointfm['num_scen']):,} next-day scenarios of "
        f"{int(jointfm['assets'])} assets from a rebalancing decision's look-back "
        "window\n(Historical replays the window's days); JointFM's call includes "
        f"the HTTP round trip to its local service.\n{_hardware_text(table)}",
    )
    _headline(figure, "Scenario generation on CPU and GPU: seconds per call")
    _style_text(figure)
    return _save(figure, directory, "runtime")


def _ratio_text(ratio: pd.Series) -> str:
    return (
        f"CPU ÷ GPU {ratio['estimate']:.1f}× "
        f"({ratio['ci_low']:.1f}–{ratio['ci_high']:.1f})"
    )


def _spread_labels(midpoints: Sequence[float], spacing: float) -> list[float]:
    """Label heights at the segment midpoints, raised bottom-up to keep ``spacing`` apart and above the baseline."""
    heights = []
    floor = spacing / 2
    for midpoint in midpoints:
        heights.append(max(midpoint, floor))
        floor = heights[-1] + spacing
    return heights


def plot_stages(table: pd.DataFrame, directory: Path) -> list[Path]:
    """One bar per device stacking a decision's scenarios and optimizer seconds (``stage_table``).

    The bars share one linear axis, since stacked segments add up only on a
    linear scale. Each segment is labeled right of its bar, the GPU's with the
    stage's CPU-to-GPU ratio; a label that does not fit at its segment's
    midpoint is raised and joined to it by a leader line. The device's tick
    label carries the decision's total, the GPU's also its ratio, and a
    whisker on each bar spans the total's interval.
    """
    figure, axes = _figure(8, 6.2)
    ax = axes[0, 0]
    seconds = table[table["reading"] == "seconds"].set_index(["stage", "device"])
    ratios = table[table["reading"] == DEVICE_RATIO].set_index("stage")
    total = seconds.loc["seconds"]
    top = total["ci_high"].max() * STACK_TOP_ROOM
    tick_labels = []
    for position, device in enumerate(DEVICE_ORDER):
        bottom = 0.0
        midpoints, texts = [], []
        for stage in STACKED_STAGES:
            row = seconds.loc[(stage, device)]
            ax.bar(
                position,
                row["estimate"],
                STACK_BAR_WIDTH,
                bottom=bottom,
                color=STAGE_COLORS[stage],
                edgecolor=SURFACE,
                linewidth=MARKER_EDGE,
            )
            name = STAGES[stage]
            if stage == "optimizer_seconds":
                name += f" ({SOLVERS_BY_NAME[row['solver']].label})"
            text = f"{name}: {_duration(row['estimate'])}"
            if device == "GPU":
                text += f"\n{_ratio_text(ratios.loc[stage])}"
            midpoints.append(bottom + row["estimate"] / 2)
            texts.append(text)
            bottom += row["estimate"]
        edge = position + STACK_BAR_WIDTH / 2
        heights = _spread_labels(midpoints, STACK_LABEL_SPACING * top)
        for midpoint, height, text in zip(midpoints, heights, texts):
            moved = not np.isclose(midpoint, height)
            ax.annotate(
                text,
                xy=(edge, midpoint),
                xytext=(edge + STACK_LABEL_OFFSET, height),
                va="center",
                color=INK_SECONDARY,
                fontsize=TEXT_SIZE,
                arrowprops={
                    "arrowstyle": "-",
                    "color": AXIS,
                    "linewidth": 1,
                    "relpos": (0, 0.5),
                }
                if moved
                else None,
            )
        ax.vlines(
            position,
            total.loc[device, "ci_low"],
            total.loc[device, "ci_high"],
            color=INK,
            linewidth=LINE_WIDTH,
        )
        label = f"{device}\n{_duration(total.loc[device, 'estimate'])} per decision"
        if device == "GPU":
            label += f"\n{_ratio_text(ratios.loc['seconds'])}"
        tick_labels.append(label)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(len(DEVICE_ORDER)), tick_labels)
    ax.set_xlim(-STACK_BAR_WIDTH, len(DEVICE_ORDER) - 1 + STACK_RIGHT_ROOM)
    ax.set_ylim(0, top)
    ax.set_ylabel("seconds per decision", color=INK_SECONDARY, fontsize=TEXT_SIZE)
    first = table.iloc[0]
    _panel_title(
        ax,
        "Seconds per rebalancing decision, by stage",
        f"mean over {int(seconds['trials'].min())} trials; the whisker spans the "
        "total's 95 % interval; decisions without warm-up",
    )
    _footnote(
        figure,
        f"A decision draws {int(first['num_scen']):,} JointFM scenarios of "
        f"{int(first['assets'])} assets and solves the Mean-CVaR problem on them;\n"
        "the scenarios include the HTTP round trip to JointFM's local service, the "
        "optimizer the problem's setup and solve.\nThe ratio is the median over "
        f"trials of CPU seconds over GPU seconds.\n{_hardware_text(table)}",
    )
    _headline(figure, "A rebalancing decision on CPU and GPU")
    _style_text(figure)
    return _save(figure, directory, "runtime_devices")


def plot_solvers(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Each paired solver's mean seconds per solve, labeled with its time over the GPU solver's."""
    figure, axes = _figure(8, SECONDS_ROW_HEIGHT * len(table) + 2.8)
    ax = axes[0, 0]
    reference = SOLVERS_BY_NAME[table["reference"].iloc[0]].label
    rows = []
    for row in table.itertuples():
        label = f"{row.label} on {row.device}"
        if row.solver != table["reference"].iloc[0]:
            label += (
                f"\n{row.ratio:.1f}× the time of {reference} "
                f"({row.ratio_ci_low:.1f}–{row.ratio_ci_high:.1f})"
            )
        rows.append(
            (label, SOLVER_COLORS[row.solver], row.estimate, row.ci_low, row.ci_high)
        )
    _seconds_rows(ax, rows, "solve without setup", SECONDS_TICK_STEPS)
    first = table.iloc[0]
    _panel_title(
        ax,
        "Seconds per solve of a rebalancing decision's problem",
        f"mean over {int(table['trials'].min())} trials with its 95 % interval;"
        " solves without warm-up",
    )
    _footnote(
        figure,
        f"Every solver solves the same {int(first['decisions'])} problems of "
        f"{int(first['num_scen']):,} scenarios of {int(first['assets'])} assets; "
        f"its weights differ from {reference}'s\nby at most "
        f"{table['max_weight_difference'].max():.1g}. The ratio is the median over "
        f"trials of the solver's seconds over {reference}'s.\n{_hardware_text(table)}",
    )
    _headline(figure, "Solving the Mean-CVaR problem on CPU and GPU")
    _style_text(figure)
    return _save(figure, directory, "solvers")


def write_device_figures(
    tables: dict[str, pd.DataFrame], directory: Path
) -> list[Path]:
    """Write each table of ``device_tables`` as CSV and draw its figure into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name in TABLES:
        path = directory / f"{name}.csv"
        tables[name].to_csv(path, index=False)
        files.append(path)
    files += plot_runtime(tables["runtime"], directory)
    files += plot_stages(tables["runtime_devices"], directory)
    files += plot_solvers(tables["solvers"], directory)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.devices_report",
        description="Tables and figures of the CPU-against-GPU experiment, written "
        "to <output stem>_report beside its results; runs nothing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration of the CPU-against-GPU experiment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    devices, _ = load_devices_config(args.config)
    tables = device_tables(devices.output)
    files = write_device_figures(tables, devices.report_dir)
    columns = ["stage", "reading", "device", "solver", "estimate", "ci_low", "ci_high"]
    print(tables["runtime_devices"][columns].to_string(index=False))
    print(f"Device report finished: {len(files)} files in {devices.report_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
