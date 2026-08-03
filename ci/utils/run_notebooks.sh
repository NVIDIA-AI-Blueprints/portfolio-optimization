#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Execute the example notebooks on a GPU and convert the results to HTML.
#
# Intended to run inside the CUDA container used by the GPU CI lanes, from the
# repo root. Both the post-merge lane (.github/workflows/main.yml) and the
# per-PR lane (.github/workflows/pr.yaml) call this so the two cannot drift --
# an inline copy in each workflow is what produced the stale release-2512.yml.
#
# Usage: ci/utils/run_notebooks.sh [CUDA_EXTRA]
#   CUDA_EXTRA  uv extra providing the GPU stack (default: cuda13)
#
# Exits non-zero if any notebook fails to execute or convert. A per-notebook
# PASS/FAIL summary is written to notebooks/notebook_status.txt.

set -euo pipefail

CUDA_EXTRA="${1:-cuda13}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# "<input stem>:<output stem>". The output stems are load-bearing: the
# artifact upload and the QA verification step in main.yml reference these
# exact filenames, and launchable deliberately maps to launchable_brev_result.
NOTEBOOKS=(
  "mean_variance_basic:mean_variance_basic_result"
  "cvar_basic:cvar_basic_result"
  "efficient_frontier:efficient_frontier_result"
  "rebalancing_strategies:rebalancing_strategies_result"
  "launchable:launchable_brev_result"
)

echo "===================== Python and GPU Info ====================="
python --version
nvidia-smi

echo "===================== Install uv ====================="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck source=/dev/null
  source "$HOME/.local/bin/env"
fi

echo "===================== Install Dependencies (extra: ${CUDA_EXTRA}) ====================="
uv sync --extra "${CUDA_EXTRA}" --extra dev

echo "===================== Install Jupyter and other packages ====================="
# Deliberately the container's `pip`, not `uv pip`: papermill and nbconvert are
# invoked as bare commands below, so they must land on PATH. `uv pip` installs
# into .venv, whose bin/ is not on PATH, giving "papermill: command not found".
# The notebooks still execute against the uv environment via the portfolio-opt
# kernel registered below.
pip install jupyter jupyterlab ipykernel papermill nbconvert

echo "===================== Create Jupyter Kernel ====================="
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"
jupyter kernelspec list

cd notebooks

overall_status=0
: > notebook_status.txt

run_notebook() {
  local input="$1"
  local output="$2"
  local name="${input%.ipynb}"

  echo "===================== Run ${input} notebook ====================="
  set +e
  # Elide base64 image payloads so the log stays readable.
  papermill "$input" "$output" --kernel portfolio-opt --log-output --log-level DEBUG 2>&1 \
    | sed -E "s/\x27image\/png\x27: \x27[^\x27]*\x27/\x27image\/png\x27: \x27<elided>\x27/g" \
    | tee "${name}.papermill.log"
  local status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -eq 0 ]; then
    echo "${input},PASS" | tee -a notebook_status.txt
  else
    echo "${input},FAIL(${status})" | tee -a notebook_status.txt
    overall_status=1
  fi
}

convert_notebook() {
  local input="$1"
  local name="${input%.ipynb}"

  if [ ! -s "$input" ]; then
    echo "nbconvert:${input},SKIP(no output notebook)" | tee -a notebook_status.txt
    overall_status=1
    return
  fi

  echo "===================== Convert ${input} to HTML ====================="
  set +e
  jupyter nbconvert --to html "$input" 2>&1 | tee "${name}.nbconvert.log"
  local status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -eq 0 ]; then
    echo "nbconvert:${input},PASS" | tee -a notebook_status.txt
  else
    echo "nbconvert:${input},FAIL(${status})" | tee -a notebook_status.txt
    overall_status=1
  fi
}

for entry in "${NOTEBOOKS[@]}"; do
  run_notebook "${entry%%:*}.ipynb" "${entry##*:}.ipynb"
done

echo "===================== Convert notebooks to HTML ====================="
for entry in "${NOTEBOOKS[@]}"; do
  convert_notebook "${entry##*:}.ipynb"
done

echo "===================== Notebook status summary ====================="
cat notebook_status.txt
exit "$overall_status"
