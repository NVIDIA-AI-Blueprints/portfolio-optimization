<!-- AI agents: Start by reading AGENTS.md -->
# Portfolio Optimization Powered by NVIDIA cuOpt

[![Build Status](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/main.yml/badge.svg)](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/main.yml)
[![Checks](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/checks.yml/badge.svg)](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/checks.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/Contributing-Guide-green)](CONTRIBUTING.md)

## Disclaimer
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

---

## Overview

This portfolio optimization developer example addresses the financial industry's trade-off between **computational speed** and **model complexity**. By leveraging **NVIDIA accelerated computing** — **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** for GPU-accelerated portfolio solves, and **[RAPIDS cuML](https://github.com/rapidsai/cuml)** for GPU scenario generation — this solution transforms robust analysis (e.g., Mean-CVaR, large-scale simulations) from slow batch processing into a **fast, iterative workflow** for dynamic decision-making.

### Accelerated Architecture

The end-to-end pipeline connects market data ingestion to optimal strategy backtesting using the NVIDIA CUDA ecosystem:

#### 1. Data Science & Scenario Generation
* **Technology:** **CUDA-X Data Science** — **[RAPIDS cuML](https://github.com/rapidsai/cuml)** for GPU KDE scenario generation
* **Function:** Accelerates data preprocessing and the learning/sampling of return distributions.
* **Performance:** Achieves speedups of up to **100x** when generating scenarios.

#### 2. Mean-CVaR Optimization
* **Technology:** **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** open-source solvers.
* **Function:** Efficiently solves complex, scenario-based **Mean-CVaR portfolio optimization** problems.
* **Performance:** Consistently outperforms state-of-the-art CPU-based solvers, with up to **160x speedups** in large-scale problems.

#### 3. Strategy Backtesting & Refinement
* **Technology:** **CUDA-X Data Science** and **HPC SDK**.
* **Function:** Rigorously tests the **trading strategies** and provides insights into strategy fine-tuning.

### Key Takeaways

* **Speed-ups:** Up to **160x faster** optimization and **100x faster** scenario generation.
* **Risk Modeling:** Enables the use of **Conditional Value-at-Risk (CVaR)** at production speed.
* **Iterative Workflow:** Supports dynamic, fast, and data-driven optimization cycles.

<p align="center">
    <img src="./docs/arch_diagram.png" alt="architecture diagram for PO" width="750"/>
</p>

---
## Get Started
### System Requirements
<details>
<summary><b>Recommended Requirements for Best Performance</b></summary>

- **System Architecture**:
  - x86-64
  - ARM64
- **GPU**:
  - NVIDIA H100 SXM (compute capability >= 9.0) and above
- **CPU**:
  - 32+ cores
- **System Memory**:
  - 64+ GB RAM
- **NVMe SSD Storage**:
  - 100+ GB free space
- **CUDA**:
  - 13.0
- **NVIDIA Drivers**:
  - Latest NVIDIA drivers (580.65.06+)
- **OS**:
  - Linux distributions with glibc>=2.28 (released in August 2018):
    - Arch Linux (minimum version 2018-08-02)
    - Debian (minimum version 10.0)
    - Fedora (minimum version 29)
    - Linux Mint (minimum version 20)
    - Rocky Linux / Alma Linux / RHEL (minimum version 8)

The above configuration will provide optimal performance for large-scale optimization problems.

</details>

### Installation on PyTorch Container

To install dependencies on the NVIDIA PyTorch container:

```bash
# Start the container. Publish 8888 for Jupyter and 8501 for Streamlit.
docker run --gpus all -it --rm \
  -v ./:/workspace/host \
  --ipc=host \
  -p 8888:8888 \
  -p 8501:8501 \
  nvcr.io/nvidia/pytorch:25.10-py3

# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization.git
cd portfolio-optimization

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# To add $HOME/.local/bin to your PATH, either restart your shell or run:
source $HOME/.local/bin/env  # (sh, bash, zsh)
# source $HOME/.local/bin/env.fish  # (fish)

# Install with CUDA-specific dependencies, plus the Jupyter kernel tooling
uv sync --extra cuda13 --group notebooks # full CUDA 13 stack currently tracks cuOpt/cuML 26.04
# On CUDA 12 hosts, use the full cuOpt/cuML 26.06 stack:
# uv sync --extra cuda12 --group notebooks
# For direct SOCP testing on CUDA 13 with cuOpt 26.06:
# uv sync --extra cuda13-socp --group notebooks

# Optional: Install development tools (ruff, pytest, pre-commit)
uv sync --extra cuda13 --group notebooks --group dev

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Launch Jupyter Lab
uv run jupyter lab --no-browser --NotebookApp.token=''
```

**Note:** If you use a different container image than the suggested one above, during uv sync, use `--extra cuda12` for the full cuOpt/cuML 26.06 CUDA 12 stack or `--extra cuda13` for the current full CUDA 13 cuOpt/cuML stack. As of the cuOpt 26.06 release, `cuml-cu13` 26.06 is not published, so CUDA 13 SOCP testing with cuOpt 26.06 uses `--extra cuda13-socp`; that extra is cuOpt-only and is intended for direct SOCP preview/Mean-Variance variance-cap solves, not GPU KDE/CVaR rebalancing. The `uv sync` command automatically creates a virtual environment and installs all dependencies from `uv.lock`.

**Tip:** To check your CUDA version, run `nvidia-smi` and look for "CUDA Version" in the output.

**Important Notes:**
- If you encounter "No space left on device" errors, set `UV_CACHE_DIR` to an alternate cache location: `export UV_CACHE_DIR=/path/to/cache/directory`
- The `cuda12`, `cuda13`, and `cuda13-socp` extras are mutually exclusive - install only one based on your system's CUDA version and workflow
- If you plan to run the Streamlit demo from this container, include `-p 8501:8501` when starting Docker. Docker port mappings cannot be added to an already-running container; restart the container with the port published if it was omitted.

#### Using the Jupyter Kernel

After launching Jupyter Lab:
1. Navigate to the [`notebooks/`](notebooks/) directory
2. Open any notebook (e.g., `cvar_basic.ipynb`)
3. Select the "Portfolio Optimization" kernel from the kernel selector in the top-right corner
4. If the kernel is not visible, refresh the page or restart Jupyter Lab

To list all available kernels:
```bash
jupyter kernelspec list
```

To remove the kernel later (if needed):
```bash
jupyter kernelspec uninstall portfolio-opt
```

### Quick Start Locally

Explore the example notebooks in the [`notebooks/`](notebooks/) directory:
- **`cvar_basic.ipynb`**: Complete walkthrough of Mean-CVaR portfolio optimization with GPU acceleration
- **`efficient_frontier.ipynb`**: A quick tutorial on how to generate efficient frontier.
- **`rebalancing_strategies.ipynb`** Introduction to dynamic re-balancing and examples of testing strategies

### Streamlit GTC Demo

The Streamlit demo from the GTC branch is available under [`demo/`](demo/) as a dynamic rebalancing app.

If you are using the PyTorch Docker container above, make sure it was started with `-p 8501:8501`. Streamlit must bind to `0.0.0.0` inside the container so the published Docker port can receive browser traffic.

```bash
uv pip install -r demo/requirements.txt
uv run python -c "from portfolio_optimization.utils import download_data; download_data('data/stock_data', datasets=['sp500'])"
uv run streamlit run demo/rebalancing_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

For a remote GPU host, also forward port `8501` from your laptop to the host running Docker:

```bash
ssh -L 8501:localhost:8501 <user>@<remote-host>
```

See [`demo/README_streamlit.md`](demo/README_streamlit.md) for focused deployment instructions.

### Deploy on Brev
Deploy using [Brev launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-360InRZzyHqDnJYQKIxaSggF8xI): start an instance on Brev.nvidia.com and follow the instructions in the notebooks.


---
## Contribution Guidelines

We welcome contributions to this project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Code of conduct
- How to submit issues and feature requests
- Pull request process
- Coding standards and best practices

---
## Community

For questions, discussions, and community support:
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/discussions)

---
## References

- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) — source repository
- [NVIDIA cuOpt Documentation](https://docs.nvidia.com/cuopt/)
- [RAPIDS cuML](https://github.com/rapidsai/cuml) — source repository
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk". *Journal of Risk*, 2, 21-42.

---
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
