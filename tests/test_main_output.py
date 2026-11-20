"""Tests for the compact command-line startup panel."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from mdtrace.console import calculation_progress, frame_range
from mdtrace.main import _print_run_summary, _show_help


class MainOutputTests(unittest.TestCase):
    def test_each_block_has_its_own_line_timer_and_completion(self):
        output = io.StringIO()
        times = [0, 10, 10, 20, 21, 26]
        with redirect_stdout(output), patch("mdtrace.console.perf_counter", side_effect=times):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with calculation_progress(4) as update:
                    update(0)
                    update(.5)  # Half of the first block, not half of the run.
                    update(.5)  # Repeated progress does not print again.
                    update(1)
                    update(1)  # Next block starts at zero on a fresh line.
                    update(1.25, frame_range(3501, 3564, 14000), 279, 500)
                    raise RuntimeError("interrupted")
        leading, first, blank, second, child, end = output.getvalue().split("\n")
        self.assertEqual(leading, "")
        self.assertEqual(first.count("Block 1/4"), 3)
        self.assertIn("50% 10.0 s elapsed · 10.0 s left", " ".join(first.split()))
        self.assertIn("100% 20.0 s elapsed", " ".join(first.split()))
        self.assertEqual(second.count("Block 2/4"), 2)
        self.assertIn("0% 0.0 s elapsed", " ".join(second.split()))
        self.assertEqual(blank, "")
        self.assertIn("25% 5.0 s elapsed · 15.0 s left", " ".join(second.split()))
        self.assertIn("└─ Q vectors", child)
        self.assertIn("55% 279/500 · frames 3501-3564 / 14000", " ".join(child.split()))
        main_line = second.rsplit("\r", 1)[-1]
        self.assertEqual(main_line.index("Block"), child.index("Q vectors"))
        self.assertEqual(main_line.index("["), child.index("["))
        self.assertEqual(main_line.index("%"), child.index("%"))
        self.assertNotIn("\x1b", output.getvalue())
        self.assertNotIn("100%", second)
        self.assertEqual(end, "")

    def test_terminal_refreshes_the_q_sub_bar_and_separates_completed_blocks(self):
        output = io.StringIO()
        with redirect_stdout(output), patch.object(output, "isatty", return_value=True), \
                patch.dict("os.environ", TERM="xterm", COLUMNS="100"), \
                patch("mdtrace.console.perf_counter", side_effect=range(30)):
            with calculation_progress(2) as update:
                for block in range(2):
                    update(block)
                    start = block * 3500
                    for fraction, stage in ((0, frame_range(start + 1, start + 64, 7000)),
                                            (.99, "FFT · " + frame_range(start + 1, start + 3500, 7000))):
                        for done in (0, 99, 1000):
                            update(block + fraction, stage, done, 1000)
                    update(block + 1)
        text = output.getvalue()
        self.assertNotIn("FFT", text)
        self.assertNotIn(",", text)
        self.assertEqual(text.count("\x1b[?25l"), 1)
        self.assertTrue(text.endswith("\x1b[?25h"))
        first, second = text.split("\n\n")
        for index, display in enumerate((first, second), start=1):
            self.assertIn(f"    ▶  Block {index}/2 [" + "=" * 30 + "] 100%", display)
            self.assertIn("\n\r    └─ Q vectors", display)
            start = (index - 1) * 3500
            self.assertIn(f"0/1000 · frames {start + 1}-{start + 64} / 7000", display)
            self.assertIn("\x1b[1A\r", display)
            self.assertIn(f"100% 1000/1000 · frames {start + 1}-{start + 3500} / 7000", " ".join(display.split()))
            rows = [row for row in display.split("\r") if "└─ Q vectors" in row]
            for marker in ("]", "/1000", "frames"):
                self.assertEqual(len({row.index(marker) for row in rows}), 1)

    def test_help_describes_dsf_and_redirects_sed_to_pysed(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaises(SystemExit):
            _show_help()

        message = output.getvalue()
        self.assertNotIn("method = sed", message)
        self.assertNotIn("action = fit", message)
        self.assertIn("device = gpu", message)
        self.assertIn("provided separately by pySED", message)
        self.assertIn("EELS remains a planned extension", message)
        self.assertIn("method = dsf", message)
        self.assertIn("density_weighting = equal", message)
        self.assertIn("current_weighting = equal", message)

    def test_run_summary_is_clear_and_aligned(self) -> None:
        params = SimpleNamespace(
            method="dsf",
            action="thinking",
            device="cpu",
            max_cores=8,
            out_files_name="SrTiO3",
        )
        output = io.StringIO()

        with redirect_stdout(output):
            _print_run_summary("input.in", params)

        message = output.getvalue()
        self.assertIn("🚀 Starting MDtrace task", message)
        self.assertIn("        Input                   : input.in", message)
        self.assertIn("        Method                  : DSF", message)
        self.assertIn("        Action                  : thinking", message)
        self.assertIn("        Device                  : CPU", message)
        self.assertIn("        Output prefix           : SrTiO3", message)

        params.device = "gpu"
        output = io.StringIO()
        with redirect_stdout(output):
            _print_run_summary("input.in", params)
        self.assertIn("        Device                  : GPU", output.getvalue())


if __name__ == "__main__":
    unittest.main()
