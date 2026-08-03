# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the portfolio optimization test suite."""

import pytest

# cuOpt gained QCQP/SOCP support in the 26.06 line. Earlier builds raise
# "Quadratic constraints not supported" from QuadraticExpression.__le__, so
# variance-cap tests must skip rather than fail against them -- the `cuda13`
# extra still tracks cuOpt 26.04 (see pyproject `cuda13` vs `cuda13-socp`).
CUOPT_SOCP_MIN_VERSION = (26, 6)


@pytest.fixture()
def require_cuopt_socp():
    """Skip unless the installed cuOpt can express quadratic constraints."""
    cuopt = pytest.importorskip("cuopt", reason="cuOpt GPU runtime required")
    raw = str(getattr(cuopt, "__version__", "0.0"))
    try:
        version = tuple(int(part) for part in raw.split(".")[:2])
    except ValueError:  # pragma: no cover - unparseable version string
        pytest.skip(f"cannot parse cuOpt version {raw!r}")
    if version < CUOPT_SOCP_MIN_VERSION:
        pytest.skip(
            f"cuOpt {raw} lacks QCQP/SOCP support; "
            f"{'.'.join(map(str, CUOPT_SOCP_MIN_VERSION))}+ required"
        )
    return version
