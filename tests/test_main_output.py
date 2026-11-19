"""Tests for the compact command-line startup panel."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from mdtrace.main import _print_run_summary, _show_help


class MainOutputTests(unittest.TestCase):
    def test_help_describes_dsf_and_redirects_sed_to_pysed(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaises(SystemExit):
            _show_help()

        message = output.getvalue()
        self.assertNotIn("method = sed", message)
        self.assertNotIn("action = fit", message)
        self.assertNotIn("backend = cupy", message)
        self.assertIn("provided separately by pySED", message)
        self.assertIn("EELS remains a planned extension", message)
        self.assertIn("method = dsf", message)

    def test_run_summary_is_clear_and_aligned(self) -> None:
        params = SimpleNamespace(
            method="dsf",
            action="thinking",
            backend="numpy",
            max_cores=8,
            out_files_name="SrTiO3",
        )
        output = io.StringIO()

        with redirect_stdout(output):
            _print_run_summary("input.in", params)

        message = output.getvalue()
        self.assertIn("🚀 Starting MDtrace task", message)
        self.assertIn("  Input   : input.in", message)
        self.assertIn("  Method  : DSF", message)
        self.assertIn("  Action  : thinking", message)
        self.assertIn(
            "  Backend : NumPy (serial DSF)",
            message,
        )
        self.assertIn("  Output  : SrTiO3", message)


if __name__ == "__main__":
    unittest.main()
