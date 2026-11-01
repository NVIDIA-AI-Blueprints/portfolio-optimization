# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text and layout every blog figure shares: names, sizes, titles, and rows of seconds.

The blog's figures come from two modules, ``blog`` for the scenario and
rebalancing results and ``devices_report`` for the CPU-against-GPU
experiment; both draw with these helpers, so every figure of the post reads
alike.
"""

from collections.abc import Sequence

from matplotlib.ticker import LogLocator, NullLocator, StrMethodFormatter

from .plotting import (
    GRID,
    INK,
    INK_SECONDARY,
    INTERVAL_MARKER_AREA,
    LINE_WIDTH,
    MARKER_EDGE,
    MUTED,
    SURFACE,
)
from .report import _duration

# The simulators as the post names them.
BLOG_NAMES = {
    "no_fit": "Historical",
    "gaussian": "Gaussian",
    "kde": "KDE",
    "jointfm": "JointFM",
}
# Sized for a figure shown about 700 px wide in a post.
HEADLINE_SIZE = 15
TITLE_SIZE = 12
TEXT_SIZE = 10.5
SUBTITLE_SIZE = 9.5
TITLE_PAD = 22
# Room left of the fastest interval and right of the slowest for its value
# label, as factors on a log axis of seconds.
SECONDS_LEFT_ROOM = 1.5
SECONDS_RIGHT_ROOM = 4.0
# Labeled steps within each decade of a log axis of seconds: 1-2-5 for an
# axis of a few decades, whole decades only for one that spans many.
SECONDS_TICK_STEPS = (1.0, 2.0, 5.0)
DECADE_TICK_STEPS = (1.0,)


def _style_text(figure) -> None:
    for ax in figure.axes:
        ax.tick_params(labelsize=TEXT_SIZE)


def _headline(figure, text: str) -> None:
    figure.suptitle(text, x=0.01, ha="left", color=INK, fontsize=HEADLINE_SIZE)


def _panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=SUBTITLE_SIZE,
        va="bottom",
    )


def _axis_labels(ax, x: str, y: str) -> None:
    ax.set_xlabel(x, color=INK_SECONDARY, fontsize=TEXT_SIZE)
    ax.set_ylabel(y, color=INK_SECONDARY, fontsize=TEXT_SIZE)


def _seconds_rows(
    ax,
    rows: Sequence[tuple[str, str, float, float, float]],
    unit: str,
    tick_steps: Sequence[float],
) -> None:
    """One row per (label, color, mean, low, high), top to bottom, on a log axis of seconds.

    Each row is a dot at its mean and a line across its interval, labeled with
    the mean right of the interval; the axis counts seconds per ``unit`` and
    labels ``tick_steps`` within each decade.
    """
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    for position, (_, color, mean, low, high) in enumerate(rows):
        ax.hlines(position, low, high, color=color, linewidth=LINE_WIDTH)
        ax.scatter(
            [mean],
            [position],
            s=INTERVAL_MARKER_AREA,
            facecolor=color,
            edgecolor=SURFACE,
            linewidth=MARKER_EDGE,
            zorder=3,
        )
        ax.annotate(
            _duration(mean),
            xy=(high, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=TEXT_SIZE,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=tick_steps))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(
        min(row[3] for row in rows) / SECONDS_LEFT_ROOM,
        max(row[4] for row in rows) * SECONDS_RIGHT_ROOM,
    )
    ax.set_xlabel(f"seconds per {unit} (log)", color=INK_SECONDARY, fontsize=TEXT_SIZE)
