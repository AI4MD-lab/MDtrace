"""DSF routing, input validation, and CLI migration tests."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from mdtrace.dsf.results import load_spectra

import numpy as np
import pytest

from mdtrace.parser import read_input
from mdtrace.pipeline import run
from mdtrace import pipeline
from mdtrace.io import NetCDFReader
from tests.io.fixtures import GPUMD_TEXT


def control(tmp_path, text):
    path = tmp_path / "input.in"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_method_is_dsf_and_has_no_sed_parameters(tmp_path):
    params = read_input(control(tmp_path, "action = plot\n"))
    assert params.method == "dsf"
    assert not hasattr(params, "basis_lattice_file")
    assert not hasattr(params, "q_path")
    assert params.device == "cpu"
    assert params.density_weighting == "equal"
    assert params.current_weighting == "equal"


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_device_input(tmp_path, device):
    params = read_input(control(tmp_path, f"action = plot\ndevice = {device}\n"))
    assert params.device == device


@pytest.mark.parametrize("text, expected", [
    ("density_weighting = equal", "equal"),
    ("density_weighting = neutron", "neutron"),
    ("experiment = number", "equal"),
    ("experiment = neutron", "neutron"),
])
def test_weighting_input_names(tmp_path, text, expected):
    params = read_input(control(tmp_path, "action = plot\n" + text + "\n"))
    assert params.density_weighting == expected
    assert not hasattr(params, "experiment")


def test_weighting_aliases_cannot_be_combined(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        read_input(control(tmp_path, "action = plot\ndensity_weighting = equal\nexperiment = number\n"))


@pytest.mark.parametrize("text", [
    "density_weighting = mass", "density_weighting = xray", "density_weighting = number",
    "current_weighting = mass", "current_weighting = neutron",
])
def test_unimplemented_weighting_cannot_silently_use_equal(tmp_path, text):
    with pytest.raises(ValueError, match="weighting"):
        read_input(control(tmp_path, "action = compute\ntime_step = 1\noutput_data_stride = 1\n" + text + "\n"))


@pytest.mark.parametrize("method", ["sed", "SED"])
def test_removed_sed_input_has_migration_message(tmp_path, method):
    with pytest.raises(ValueError, match="SED has been removed.*pySED"):
        read_input(control(tmp_path, f"method = {method}\n"))


def test_direct_pipeline_cannot_run_sed():
    with pytest.raises(ValueError, match="SED has been removed.*pySED"):
        run(SimpleNamespace(method="sed", action="compute"))


@pytest.mark.parametrize("text, message", [
    ("action plot", "expected 'key = value'"),
    ("action = plot\nnum_atoms = 10", "parameter 'num_atoms' is not valid"),
    ("action = fit", "action must be thinking, compute, or plot"),
    ("action = plot\nnum_blocks = 2 extra", "requires exactly 1 value"),
    ("action = plot\naction = compute", "parameter 'action' is repeated"),
    ("action = plot\nif_show_figures = 2", "must be 0 or 1"),
    ("action = plot\nbackend = numpy", "parameter 'backend' is not valid"),
    ("action = plot\nbackend = cupy", "parameter 'backend' is not valid"),
    ("action = plot\ndevice = numpy", "device must be cpu or gpu"),
    ("action = plot\ndevice = cupy", "device must be cpu or gpu"),
    ("action = plot\nnetcdf_batch_size = 0", "netcdf_batch_size must be positive"),
    ("action = plot\nnetcdf_compression_level = 10", "netcdf_compression_level"),
    ("action = plot\ntrajectory_read_mode = parallel", "trajectory_read_mode must be"),
    ("action = plot\ndsf_window = hann", "parameter 'dsf_window' is not valid"),
    ("action = compute\ntime_step = 0\noutput_data_stride = 1", "time_step must be positive"),
])
def test_invalid_configuration_is_rejected(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_input(control(tmp_path, text + "\n"))


def test_eels_remains_planned(tmp_path):
    params = read_input(control(tmp_path, "method = eels\n"))
    with pytest.raises(SystemExit) as error:
        run(params)
    assert error.value.code == 1


@pytest.mark.parametrize("action", ["compute", "plot"])
def test_explicit_action_runs_even_when_results_exist(tmp_path, monkeypatch, action):
    params = read_input(control(tmp_path, "action = plot\n"))
    params.action = action
    compute, plot = Mock(), Mock()
    monkeypatch.setitem(pipeline.STEP_MAP, "dsf", {"compute": compute, "plot": plot})
    monkeypatch.setattr(pipeline, "_dsf_ready", lambda *a, **k: True)
    run(params)
    (compute if action == "compute" else plot).assert_called_once_with(params)
    (plot if action == "compute" else compute).assert_not_called()


def test_thinking_recomputes_changed_inputs_then_plots(tmp_path, monkeypatch):
    params = SimpleNamespace(method="dsf", action="thinking")
    calls = []
    monkeypatch.setattr(pipeline, "_dsf_ready", lambda p, check_settings=False: not check_settings)
    monkeypatch.setitem(pipeline.STEP_MAP, "dsf", {
        "compute": lambda p: calls.append("compute"), "plot": lambda p: calls.append("plot")})
    run(params)
    assert calls == ["compute", "plot"]


@pytest.mark.parametrize("figure_state", ["missing", "outdated", "current", "changed_panels", "old_layout"])
def test_thinking_reuses_text_and_checks_figure_time(tmp_path, monkeypatch, figure_state):
    from PIL import Image, PngImagePlugin
    from mdtrace.dsf.results import plot_signature

    prefix = tmp_path / "spectrum"
    saved_text = Path(str(prefix) + ".dsf_Qaxis")
    saved_text.write_text('# {"text_format_version": 2, "metadata": {"experiment": "number", "currents": true}}\n')
    monkeypatch.setattr(pipeline, "result_files", lambda prefix, **selection: [saved_text])
    os.utime(saved_text, (20, 20))
    params = SimpleNamespace(method="dsf", action="thinking", out_files_name=str(prefix))
    for suffix in (".dsf.png", ".dsf_Ctotal.png", ".dsf_CL.png", ".dsf_CT.png"):
        if suffix == ".dsf_CT.png" and figure_state == "missing":
            continue
        figure = Path(str(prefix) + suffix)
        info = PngImagePlugin.PngInfo()
        signature = "{}" if suffix == ".dsf.png" and figure_state == "old_layout" else plot_signature(params)
        info.add_text("MDtrace plot", signature)
        Image.new("RGB", (2, 2)).save(figure, pnginfo=info)
        os.utime(figure, (10, 10) if suffix == ".dsf_CL.png" and figure_state == "outdated" else (30, 30))
    if figure_state == "changed_panels":
        params.dsf_plot_density = False
    compute, plot = Mock(), Mock()
    monkeypatch.setitem(pipeline.STEP_MAP, "dsf", {"compute": compute, "plot": plot})
    monkeypatch.setattr(pipeline, "_dsf_ready", lambda *a, **k: True)
    run(params)
    compute.assert_not_called()
    assert plot.call_count == (0 if figure_state == "current" else 1)


def cli(tmp_path, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, "-B", "-m", "mdtrace.main", *map(str,args)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")


def test_cli_default_dsf_computes_and_replots_without_source(tmp_path):
    trajectory = tmp_path / "dump.xyz"
    trajectory.write_text(GPUMD_TEXT, encoding="utf-8")
    path = control(tmp_path, "action = compute\nexperiment = number\ntime_step = 1\n"
                   "output_data_stride = 1\nnum_blocks = 1\n"
                   "dsf_q_coordinates = cartesian\ndsf_qpoints = 0 0 0\n")
    completed = cli(tmp_path, path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    saved_text = tmp_path / "mdtrace.dsf_Qaxis"
    assert saved_text.is_file()
    result = load_spectra(str(tmp_path / "mdtrace"))
    assert np.isfinite(result["sqw"]).all()
    np.testing.assert_allclose(result["freq_thz"], [-500., 0.])
    cache = tmp_path / "dump.xyz.mdtrace.nc"
    assert cache.is_file()
    with NetCDFReader(cache) as reader:
        assert reader.info.n_frames == 2
    before = saved_text.read_bytes()
    trajectory.unlink()  # only this test's synthetic source
    cache.unlink()
    path.write_text("action = plot\n", encoding="utf-8")
    plotted = cli(tmp_path, path)
    assert plotted.returncode == 0, plotted.stdout + plotted.stderr
    assert (tmp_path / "mdtrace.dsf.png").is_file()
    assert saved_text.read_bytes() == before


def test_legacy_sed_input_is_not_auto_selected_or_modified(tmp_path):
    legacy = tmp_path / "input_SED.in"
    legacy.write_text("method = sed\n", encoding="utf-8")
    result = cli(tmp_path)
    assert result.returncode == 1
    assert "No input.in found" in result.stdout
    explicit = cli(tmp_path, legacy)
    assert explicit.returncode == 1
    assert "SED has been removed" in explicit.stderr and "pySED" in explicit.stderr
    assert "Traceback" not in explicit.stderr
    assert legacy.read_text(encoding="utf-8") == "method = sed\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input_SED.in"]
