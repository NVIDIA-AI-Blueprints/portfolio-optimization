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

set -euo pipefail

CUDA_EXTRA="${1:-cuda13}"
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

echo "===================== Run GPU test suite ====================="
# -rs surfaces skip reasons, so a lane that silently skips everything
# (wrong extra, missing cuML) is visible in the log instead of reading green.
uv run pytest tests/ -v -m gpu -rs
