# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the CPU-against-GPU experiment in experiments/: the stage clock,
the sessions' rows and resumption, the environment checks, and the report's
tables, with the fake JointFM client and the device stand-ins of conftest.py."""

import os
import platform
import socket

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.simulator_comparison import devices as devices_module, devices_report
from experiments.simulator_comparison.config import DevicesConfig, ExperimentConfig
from experiments.simulator_comparison.devices import (
    SERVICE_ENVIRONMENT,
    foreign_cores,
    run_decisions,
    run_device_session,
    select_session_gpus,
    service_gpu,
    service_process,
    solve_decision,
)
from experiments.simulator_comparison.devices_report import (
    DEVICE_RATIO,
    device_tables,
    read_device_results,
    write_device_figures,
)
from experiments.simulator_comparison.rebalancing import (
    calibrated_cvar_params,
    draw_rebalancing_trial,
    write_universe,
)
from experiments.simulator_comparison.runner import load_prices
from experiments.simulator_comparison.table import (
    RunMetadata,
    read_results,
    write_results,
)
from portfolio_optimization import risk_free

ROWS = 300
ASSETS = 3
STEP_DAYS = 5
STEPS = 2
TRIALS = 2
PAIRED = ("CUOPT", "HIGHS", "CLARABEL")


def _write_inputs(directory):
    rng = np.random.default_rng(11)
    dates = pd.date_range("2019-01-01", periods=ROWS, freq="B")
    # Positive drift keeps the risk-aversion calibration defined.
    log_prices = np.cumsum(rng.normal(0.001, 0.01, size=(ROWS, 6)), axis=0)
    prices = directory / "prices.csv"
    pd.DataFrame(
        100.0 * np.exp(log_prices), index=dates, columns=[f"A{i}" for i in range(6)]
    ).to_csv(prices)
    rates = directory / "rates.csv"
    pd.Series(
        0.02, index=pd.date_range("2018-12-03", "2020-06-30", freq="B"), name="rate"
    ).rename_axis("Date").to_csv(rates)
    return prices, rates


def _configs(directory, **devices):
    prices, rates = _write_inputs(directory)
    experiment = ExperimentConfig(
        prices=prices,
        output=directory / "results.parquet",
        trials=TRIALS,
        origins=2,
        assets=ASSETS,
        look_back=64,
        origin_history=64,
        forward=8,
        num_scen=50,
        seed=5,
        arms=("no_fit-cpu",),
        portfolio_sizes=(1, ASSETS),
        horizons=(1,),
        rebalancing=dict(
            output=directory / "rebalancing.parquet",
            scenario_results=directory / "results.parquet",
            arms=("no_fit-cpu",),
            trials=TRIALS,
            trading_days=STEPS * STEP_DAYS,
            look_forward_window=STEP_DAYS,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=rates,
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
            arms=("no_fit-cpu",),
            sizes=(ASSETS,),
        ),
    )
    values = dict(
        experiment=directory / "experiment.yaml",
        output=directory / "devices.parquet",
        trials=TRIALS,
        num_scen=50,
        gpu_solver="CUOPT",
        cpu_solver="HIGHS",
        paired_solvers=PAIRED,
        generators=("no_fit-cpu", "gaussian-cpu"),
    )
    values.update(devices)
    return DevicesConfig(**values), experiment


@pytest.fixture(scope="module")
def sessions(tmp_path_factory, device_sessions):
    """Both sessions of a small experiment, run once per module."""
    device_config, experiment = _configs(tmp_path_factory.mktemp("devices"))
    device_sessions(device_config, experiment)
    frame, metadata = read_results(device_config.output)
    return device_config, experiment, frame, metadata.identity


def test_every_decision_of_both_sessions_is_timed_by_stage(sessions):
    _, _, frame, _ = sessions
    decisions = frame[frame["kind"] == "decision"]
    assert len(decisions) == 2 * TRIALS * STEPS
    assert set(decisions.groupby(["session", "trial"]).size()) == {STEPS}
    np.testing.assert_allclose(
        decisions["seconds"],
        decisions["generation_seconds"] + decisions["optimizer_seconds"],
    )
    assert (decisions["optimizer_seconds"] > 0).all()
    solvers = decisions.groupby("session")["solver"].unique()
    assert solvers.to_dict() == {"CPU": ["HIGHS"], "GPU": ["CUOPT"]}


def test_each_session_times_its_own_devices_simulators(sessions):
    _, _, frame, _ = sessions
    calls = frame[frame["kind"] == "generation"]
    names = calls.groupby("session")["name"].unique().map(sorted).to_dict()
    assert names == {
        "CPU": ["gaussian-cpu", "jointfm-cpu", "no_fit-cpu"],
        "GPU": ["jointfm-gpu"],
    }
    pipeline = calls[calls["name"] == "jointfm-gpu"].set_index(["trial", "decision"])
    decided = frame[(frame["kind"] == "decision") & (frame["session"] == "GPU")]
    np.testing.assert_allclose(
        pipeline["seconds"],
        decided.set_index(["trial", "decision"])["generation_seconds"],
    )


def test_only_the_gpu_session_solves_the_paired_problems(sessions):
    device_config, _, frame, _ = sessions
    solves = frame[frame["kind"] == "solve"]
    assert set(solves["session"]) == {"GPU"}
    assert len(solves) == TRIALS * STEPS * len(PAIRED)
    assert set(solves["num_scen"]) == {device_config.num_scen}


def test_the_first_row_of_each_kind_and_name_per_session_is_warmup(sessions):
    _, _, frame, _ = sessions
    flagged = frame[frame["warmup"]]
    assert not flagged.duplicated(["session", "kind", "name"]).any()
    expected = frame.drop_duplicates(["session", "kind", "name"]).index
    assert sorted(flagged.index) == sorted(expected)


def test_the_paired_solvers_agree_on_identical_problems(sessions):
    """The GPU pipeline's stand-in is Clarabel, so it matches Clarabel exactly."""
    _, _, frame, _ = sessions
    solves = frame[frame["kind"] == "solve"]
    by_solver = {
        name: np.stack(rows.sort_values(["trial", "decision"])["weights"].to_numpy())
        for name, rows in solves.groupby("name")
    }
    np.testing.assert_allclose(by_solver["CUOPT"], by_solver["CLARABEL"])
    # A linear program can have several optimal portfolios, but one optimum.
    objectives = solves.pivot_table(
        index=["trial", "decision"], columns="name", values="objective"
    )
    np.testing.assert_allclose(objectives["HIGHS"], objectives["CLARABEL"], atol=1e-6)


def test_a_rebuilt_problem_reproduces_the_backtests_decision(tmp_path, fake_jointfm):
    """Each decision optimized from the portfolio its predecessor chose."""
    fake_jointfm.envelope.n_input = 1000
    fake_jointfm.sample_scale = 0.01
    device_config, experiment = _configs(tmp_path)
    prices = load_prices(experiment.prices)
    trial, window = draw_rebalancing_trial(prices, 0, experiment)
    universe = write_universe(prices, trial, tmp_path / "universes")
    params = calibrated_cvar_params(
        prices[list(trial.assets)], window, experiment.rebalancing
    )
    rates = risk_free.load_risk_free_rates(experiment.rebalancing.risk_free_rates)
    arm, _ = devices_module.check_jointfm_service()
    decisions = run_decisions(
        arm, 0, params, experiment, device_config, universe, window, rates, "CLARABEL"
    )
    assert len(decisions) == STEPS
    assert decisions[0].previous is None
    assert decisions[1].previous is decisions[0].chosen
    for decision in decisions:
        solved = solve_decision(decision, params, "CLARABEL")
        np.testing.assert_allclose(
            solved["weights"], decision.chosen.weights, atol=1e-7
        )
        assert solved["cash"] == pytest.approx(decision.chosen.cash, abs=1e-7)
        assert "cvar_data" not in decision.window


def test_a_rerun_resumes_without_adding_rows(sessions, device_sessions):
    device_config, experiment, frame, _ = sessions
    device_sessions(device_config, experiment)
    again, _ = read_results(device_config.output)
    assert len(again) == len(frame)


def test_a_session_refuses_a_service_on_the_other_device(sessions, device_sessions):
    device_config, experiment, _, _ = sessions
    # device_sessions' fake service reports the CPU outside a session run.
    with pytest.raises(ValueError, match="the GPU session needs one on the GPU"):
        run_device_session(device_config, experiment, "GPU")


def test_the_cpu_session_must_hide_every_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="must hide every GPU"):
        select_session_gpus("CPU")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="must hide every GPU"):
        select_session_gpus("CPU")


def test_a_services_gpu_counts_through_its_visible_devices():
    assert service_gpu("cuda", None) == "0"
    assert service_gpu("cuda:1", None) == "1"
    assert service_gpu("cuda:1", "3, 5") == "5"
    with pytest.raises(ValueError, match="sees only"):
        service_gpu("cuda:2", "0,1")


def test_a_remote_service_is_refused():
    with pytest.raises(ValueError, match="no local service"):
        service_process("https://jointfm.example.com:443")


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="psutil lists listening sockets without root on Linux only",
)
def test_the_local_service_process_is_found_by_its_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        pid, record = service_process(f"http://127.0.0.1:{port}")
    assert pid == os.getpid()
    assert list(record["environment"]) == list(SERVICE_ENVIRONMENT)
    assert record["usable_cpus"] >= 1
    assert record["command"]
    with pytest.raises(ValueError, match="Expected one process"):
        service_process(f"http://127.0.0.1:{port}")


def test_foreign_cores_count_new_processes_whole_and_drop_exited_ones():
    before = {1: 1.0, 2: 5.0}
    after = {1: 3.0, 3: 2.0}
    assert foreign_cores(before, after, 2.0) == pytest.approx(2.0)


def test_runtime_reads_every_simulator_on_its_device(sessions):
    device_config, _, frame, _ = sessions
    table = device_tables(device_config.output)["runtime"]
    assert list(table["arm"]) == [
        "no_fit-cpu",
        "gaussian-cpu",
        "jointfm-cpu",
        "jointfm-gpu",
    ]
    calls = frame[(frame["kind"] == "generation") & ~frame["warmup"]]
    per_trial = calls.groupby(["name", "trial"])["seconds"].mean()
    np.testing.assert_allclose(
        table.set_index("arm")["estimate"],
        per_trial.groupby(level=0).mean()[table["arm"]],
    )
    assert set(table["gpu"]) == {"Test GPU"}


def test_stage_ratios_are_medians_of_per_trial_ratios(sessions):
    device_config, _, frame, _ = sessions
    table = device_tables(device_config.output)["runtime_devices"]
    decisions = frame[(frame["kind"] == "decision") & ~frame["warmup"]]
    per_trial = decisions.groupby(["session", "trial"])["seconds"].mean()
    total = table[(table["stage"] == "seconds") & (table["reading"] == DEVICE_RATIO)]
    assert total["estimate"].item() == pytest.approx(
        np.median(per_trial["CPU"] / per_trial["GPU"])
    )
    seconds = table[table["reading"] == "seconds"]
    assert set(zip(seconds["device"], seconds["solver"])) == {
        ("CPU", "HIGHS"),
        ("GPU", "CUOPT"),
    }


def test_solvers_compare_with_the_gpu_pipelines_solver(sessions):
    device_config, _, _, _ = sessions
    tables = device_tables(device_config.output)
    solvers = tables["solvers"].set_index("solver")
    assert list(solvers.index) == list(PAIRED)
    assert solvers.loc["CUOPT", "ratio"] == pytest.approx(1.0)
    assert solvers.loc["CUOPT", "max_weight_difference"] == 0.0
    assert solvers.loc["HIGHS", "max_objective_gap"] < 1e-6
    assert set(solvers["decisions"]) == {TRIALS * STEPS}


def test_a_single_session_is_refused(sessions, tmp_path):
    _, _, frame, identity = sessions
    only_gpu = {
        **identity,
        "arms": {
            name: arm
            for name, arm in identity["arms"].items()
            if arm["device"] == "GPU"
        },
    }
    path = tmp_path / "gpu_only.parquet"
    write_results(
        path,
        frame[frame["session"] == "GPU"],
        RunMetadata(identity=only_gpu, commands=[]),
    )
    with pytest.raises(ValueError, match="holds no \\['CPU'\\] session"):
        read_device_results(path)


def test_every_text_of_the_device_figures_lies_inside_the_figure(
    sessions, monkeypatch, tmp_path
):
    """No title, subtitle, label, or footnote of a speed figure runs past its edges."""
    device_config, _, _, _ = sessions
    figures = {}

    def record(figure, directory, name):
        figures[name] = figure
        return []

    monkeypatch.setattr(devices_report, "_save", record)
    write_device_figures(device_tables(device_config.output), tmp_path)
    assert set(figures) == {"runtime", "runtime_devices", "solvers"}
    for name, figure in figures.items():
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        drawn = figure.get_tightbbox(canvas.get_renderer())
        assert drawn.x0 >= 0 and drawn.y0 >= 0, name
        assert drawn.x1 <= figure.get_figwidth(), name
        assert drawn.y1 <= figure.get_figheight(), name
