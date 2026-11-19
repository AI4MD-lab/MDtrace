"""Ensure analysis helpers can be used without loading trajectory computation."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("statement", [
    "import mdtrace.dsf.spectra",
    "import mdtrace.dsf.results",
    "import mdtrace.dsf.postprocess",
    "import mdtrace.dsf.plot",
    "from mdtrace.dsf import save_dsf, load_spectra",
])
def test_result_and_numerical_modules_do_not_load_compute_or_plotting(statement):
    code = statement + "\n" + """
import sys
assert 'mdtrace.dsf.dsf' not in sys.modules
assert 'mdtrace.io.netcdf' not in sys.modules
assert 'matplotlib.pyplot' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_result_entry_points_remain_compatible():
    from mdtrace.dsf import compute_dsf, load_spectra, save_dsf
    from mdtrace.dsf.dsf import compute_dsf as calculate, save_dsf as old_save
    from mdtrace.dsf.postprocess import load_spectra as old_load
    from mdtrace.dsf.results import load_spectra as load, save_dsf as save

    assert compute_dsf is calculate
    assert save_dsf is save is old_save
    assert load_spectra is load is old_load
