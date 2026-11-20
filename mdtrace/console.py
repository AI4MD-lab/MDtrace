# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Block progress and a nested counter for completed Q vectors."""

from contextlib import contextmanager
import os
from shutil import get_terminal_size
import sys
from time import perf_counter


def _bar(done, total, cells):
    filled = int(cells * done / total)
    return "[" + "=" * filled + "-" * (cells - filled) + "]"


def frame_range(first, last, total):
    return f"frames {first}-{last} / {total}"


@contextmanager
def calculation_progress(num_blocks):
    """Redraw the active rows in a terminal and keep plain summaries in logs."""
    terminal = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    width = max(1, get_terminal_size().columns - 1)
    label_width = len(f"▶  Block {num_blocks}/{num_blocks}") + 1
    # Reserve space once so changes in the counts never resize the bars.
    cells = min(30, max(1, width - label_width - 55)) if terminal else 30
    block = 0
    started = None
    last_percent, line_width = -1, 0
    last_stage, last_draw = "", 0.0
    child, child_visible = "", False

    def update(completed, stage="", q_done=0, q_total=0):
        nonlocal block, started, last_percent, line_width, last_stage, last_draw, child, child_visible
        if block >= num_blocks:
            return
        fraction = min(max(completed - block, 0.0), 1.0)
        percent = int(100 * fraction)
        now = perf_counter()
        if stage:
            if stage == last_stage and q_done < q_total and now - last_draw < 0.1 and percent == last_percent:
                return
            count = f"{q_done:>{len(str(q_total))}}/{q_total}"
            suffix = f" {100 * q_done // q_total:3d}%   {count} · {stage.removeprefix('FFT · ')}"
            child = f"    {'└─ Q vectors':<{label_width}}{_bar(q_done, q_total, cells)}{suffix}"
            last_stage = stage
        if percent == last_percent and not (terminal and stage):
            return
        if started is None:
            if terminal and block == 0:
                print("\x1b[?25l", end="")  # Hide the cursor while the two rows refresh.
            print()
            started = now
        last_percent, last_draw = percent, now
        elapsed = now - started
        label = f"▶  Block {block + 1}/{num_blocks}"
        line = f"    {label:<{label_width}}{_bar(percent, 100, cells)} {percent:3d}%   {elapsed:7.1f} s elapsed"
        if 0 < fraction < 1:
            line += f" · {elapsed * (1 - fraction) / fraction:7.1f} s left"
        if terminal:
            # The cursor stays on the child row until the next refresh.
            output = ("\x1b[1A" if child_visible else "") + "\r" + line[:width] + "\x1b[K"
            if child:
                output += ("\x1b[1B\r" if child_visible else "\n\r") + child[:width] + "\x1b[K"
            child_visible = bool(child)
        else:
            output = "\r" + line.ljust(line_width)
            if child and percent == 100:
                output += "\n" + child
        print(output, end="\n" if percent == 100 else "", flush=True)
        line_width = len(line)
        if percent == 100:
            block += 1
            started = None
            last_percent, line_width = -1, 0
            child, last_stage, child_visible = "", "", False

    try:
        yield update
    finally:
        if terminal and (started is not None or block):
            print("\x1b[?25h", end="", flush=True)
        if line_width:
            if child and not terminal:
                print("\n" + child, end="")
            print(flush=True)
