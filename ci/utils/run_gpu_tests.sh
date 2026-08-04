#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the GPU-marked pytest suite inside the CUDA container used by the GPU
# CI lanes, from the repo root. Shared by the post-merge lane
# (.github/workflows/main.yml) and the per-PR lane (.github/workflows/pr.yaml)
# so the two cannot drift.
#
# Usage: ci/utils/run_gpu_tests.sh [CUDA_EXTRA]
#   CUDA_EXTRA  uv extra providing the GPU stack (default: cuda13)
#
# Note on coverage: the `cuda13` extra tracks cuOpt 26.04, which has no
# QCQP/SOCP support, so the variance-cap tests skip themselves there (see the
# require_cuopt_socp fixture in tests/conftest.py). Use `cuda12` or
# `cuda13-socp` to exercise those paths.
#
# When CUDA_EXTRA is a SOCP-capable extra, the installed cuOpt is asserted to
# be >= 26.06 before the suite runs. Without that assertion a misresolved
# environment would skip every SOCP test and still exit 0, so the lane would
# report green while testing nothing.

set -euo pipefail

CUDA_EXTRA="${1:-cuda13}"
SOCP_MIN="26.6"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "===================== GPU Info ====================="
nvidia-smi

echo "===================== Install uv ====================="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

echo "===================== Install Dependencies (extra: ${CUDA_EXTRA}) ====================="
uv sync --extra "${CUDA_EXTRA}" --extra dev

echo "===================== Solver versions ====================="
uv run python -c "
import cuopt
print('cuopt', cuopt.__version__)
try:
    import cuml
    print('cuml', cuml.__version__)
except ImportError:
    print('cuml NOT installed - skill benchmark tests will skip')
"

case "${CUDA_EXTRA}" in
  *socp* | cuda12)
    echo "===================== Assert SOCP-capable cuOpt ====================="
    if ! uv run python -c "
import sys, cuopt
want = tuple(int(p) for p in '${SOCP_MIN}'.split('.'))
got = tuple(int(p) for p in cuopt.__version__.split('.')[:2])
print(f'cuopt {cuopt.__version__} (need >= ${SOCP_MIN})')
sys.exit(0 if got >= want else 1)
"; then
      echo "ERROR: extra '${CUDA_EXTRA}' is meant to exercise SOCP but resolved a"
      echo "       cuOpt older than ${SOCP_MIN}. The SOCP tests would skip and this"
      echo "       lane would pass without testing anything."
      exit 1
    fi
    ;;
esac

echo "===================== Run GPU test suite ====================="
# -rs surfaces skip reasons, so a lane that silently skips everything
# (wrong extra, missing cuML) is visible in the log instead of reading green.
uv run pytest tests/ -v -m gpu -rs
