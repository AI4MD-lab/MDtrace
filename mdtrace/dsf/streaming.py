# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded trajectory batches and one-item prefetch for density/current DSF."""

from concurrent.futures import as_completed, wait
from contextlib import ExitStack, contextmanager
from multiprocessing import shared_memory
from time import perf_counter
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
from pySED.io.trajectory import prefetch_one
from pySED.backends.common import available_cpu_memory, qpoint_batches
from pySED.console import print_detail, print_section

from mdtrace.console import frame_range
from mdtrace.io import NetCDFReader, POSITIONS, VELOCITIES
from mdtrace.dsf.parallel import _worker_count
from mdtrace.dsf.spectra import density_current_signals, spectra_from_signals


@contextmanager
def _buffer(shape, dtype, shared):
    """Own an array for the whole run; workers attach only while using it."""
    if not shared:
        yield None, np.empty(shape, dtype=dtype)
        return
    segment = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np.dtype(dtype).itemsize)
    try:
        yield (segment.name, shape, np.dtype(dtype).str), np.ndarray(shape, dtype=dtype, buffer=segment.buf)
    finally:
        segment.close()
        segment.unlink()


@contextmanager
def _attach(descriptor, readonly=False):
    name, shape, dtype = descriptor
    segment = shared_memory.SharedMemory(name=name)
    try:
        array = np.ndarray(shape, dtype=dtype, buffer=segment.buf)
        if readonly:
            array.setflags(write=False)
        yield array
    finally:
        segment.close()


def _fill_signals(inputs, signals, size, offset, start, stop, q_cart, coherent):
    with _attach(inputs, readonly=True) as data, _attach(signals) as output:
        density_current_signals(
            data[0, :size], q_cart[start:stop], coherent,
            output[start:stop, offset:offset + size],
            velocities=data[1, :size] if len(data) == 2 else None,
        )
    return stop - start


def _finish_signals(signals, start, stop, q_cart, n_atoms, options):
    with _attach(signals, readonly=True) as values:
        return start, stop, spectra_from_signals(values[start:stop], q_cart[start:stop], n_atoms, **options)


def _completed(pool, function, arguments):
    """Drain every submitted task before the input buffer can be overwritten."""
    futures = []
    try:
        for args in arguments:
            futures.append(pool.submit(function, *args))
        for future in as_completed(futures):
            yield future.result()
    finally:
        for future in futures:
            future.cancel()
        wait(futures)


def _batch_plan(n_atoms, block_size, n_q, currents, requested, prefetch):
    """Bound scratch arrays, raw read/conversion buffers, signals and results."""
    batch = min(64, block_size)
    fields = 2 if currents else 1
    signals = n_q * block_size * (4 if currents else 1) * 16
    outputs = n_q * block_size * (5 if currents else 3) * 8 * 3
    available = available_cpu_memory()
    if available is None:
        batch, prefetch = min(batch, 32), False
    # Reduce batch size before disabling prefetch. Even serial mode retains
    # full-block signals, but never a full-block atom trajectory or phase.
    while True:
        inputs = fields * batch * n_atoms * 3 * 8
        reads = inputs * (3 + int(prefetch))
        working = batch * n_atoms * 40 + 128 * 1024**2
        fits = available is None or signals + outputs + inputs + reads + working < available * .8
        if fits:
            break
        if batch > 1:
            batch = max(1, batch // 2)
        elif prefetch:
            prefetch = False
        else:
            raise MemoryError("Insufficient available RAM for the complete DSF time signals and spectra")
    workers = _worker_count(requested, n_q, inputs + signals, outputs + reads, working)
    return batch, workers, prefetch


def _trajectory_batches(path, block_size, num_blocks, batch_size, currents, atom_order):
    # This generator's next() runs exclusively in the reader thread when
    # prefetched. No computation worker or concurrent parent uses NetCDF.
    with NetCDFReader(path) as reader:
        for block in range(num_blocks):
            for offset in range(0, block_size, batch_size):
                start = block * block_size + offset
                frames = slice(start, start + min(batch_size, block_size - offset))
                positions = reader.read(POSITIONS, frames)
                velocities = reader.read(VELOCITIES, frames) / 100.0 if currents else None
                if atom_order is not None:
                    order = atom_order[frames, :, None]
                    positions = np.take_along_axis(positions, order, axis=1)
                    if velocities is not None:
                        velocities = np.take_along_axis(velocities, order, axis=1)
                yield positions, velocities
                del positions, velocities


def _q_counter(status, block_progress, stage, total):
    """Count finished Q vectors in the main process for one frame batch or FFT."""
    if status is None:
        return None
    done = 0

    def advance(count):
        nonlocal done
        done += count
        status(block_progress, stage, done, total)

    advance(0)
    return advance


def stream_blocks(executor, path, block_size, num_blocks, n_atoms, q_cart, coherent,
                  *, currents=False, prefetch=True, atom_order=None, progress=None,
                  batch_status=None, trajectory_frames=None, **options):
    """Yield complete spectra, with unchanged statistical-block boundaries.

    Only use when no atom-wise self/incoherent/partial FFTs are requested.
    A persistent shared signal array has disjoint Q slices for workers. The
    full time series is transformed only after all frame batches are filled.
    Close this generator on early exit so reads, tasks and buffers are drained.
    """
    requested_prefetch = prefetch
    batch, workers, prefetch = _batch_plan(
        n_atoms, block_size, len(q_cart), currents, executor.max_cores, prefetch,
    )
    input_shape = (2 if currents else 1, batch, n_atoms, 3)
    signal_shape = (len(q_cart), block_size, 4 if currents else 1)
    executor._configure(workers)
    executor.stream_wait_seconds = 0.0
    executor.stream_batch_frames = batch
    executor.stream_prefetch = prefetch
    with ExitStack() as stack:
        try:
            input_desc, data = stack.enter_context(_buffer(input_shape, np.float64, workers > 1))
            signal_desc, signals = stack.enter_context(_buffer(signal_shape, complex, workers > 1))
        except (OSError, MemoryError):
            stack.close()
            warnings.warn("Shared storage unavailable; using streamed serial DSF", RuntimeWarning)
            workers = 1
            executor._configure(1)
            input_desc, data = stack.enter_context(_buffer(input_shape, np.float64, False))
            signal_desc, signals = stack.enter_context(_buffer(signal_shape, complex, False))
        print_section("Trajectory streaming", "🚀")
        print_detail("Batch size", f"{batch} frames")
        print_detail("Prefetch", "1 (one batch ahead)" if prefetch else
                     "0 (memory availability)" if requested_prefetch else "0")
        n_q = len(q_cart)
        ranges = qpoint_batches(n_q, workers)
        if trajectory_frames is None:
            trajectory_frames = block_size * num_blocks
        source = _trajectory_batches(path, block_size, num_blocks, batch, currents, atom_order)
        batches = stack.enter_context(prefetch_one(source, enabled=prefetch))
        # Also bound parent BLAS work (serial calculation and final assembly).
        stack.enter_context(threadpool_limits(limits=1))
        for block in range(num_blocks):
            block_start = block * block_size
            if progress is not None:
                progress(block)
            for offset in range(0, block_size, batch):
                before = perf_counter()
                positions, velocities = next(batches)
                executor.stream_wait_seconds += perf_counter() - before
                size = len(positions)
                expected = min(batch, block_size - offset)
                if positions.shape != (expected, n_atoms, 3):
                    raise ValueError("trajectory batch does not match the requested frames and atoms")
                if currents and velocities.shape != positions.shape:
                    raise ValueError("velocities must match positions (frame, atom, 3)")
                np.copyto(data[0, :size], positions)
                if currents:
                    np.copyto(data[1, :size], velocities)
                del positions, velocities
                first_frame = block_start + offset + 1
                stage = frame_range(first_frame, first_frame + size - 1, trajectory_frames)
                advance = _q_counter(batch_status, block + min(offset / block_size, .99), stage, n_q)
                if workers == 1:
                    density_current_signals(data[0, :size], q_cart, coherent,
                                            signals[:, offset:offset + size],
                                            velocities=data[1, :size] if currents else None, q_completed=advance)
                else:
                    tasks = ((input_desc, signal_desc, size, offset, a, b, q_cart, coherent) for a, b in ranges)
                    for count in _completed(executor.pool, _fill_signals, tasks):
                        if advance is not None:
                            advance(count)
                if progress is not None:
                    # Leave completion to the caller after FFT and averaging.
                    progress(block + min((offset + size) / block_size, .99))
            stage = "FFT · " + frame_range(block_start + 1, block_start + block_size, trajectory_frames)
            advance = _q_counter(batch_status, block + .99, stage, n_q)
            if workers == 1:
                result = spectra_from_signals(signals, q_cart, n_atoms, q_completed=advance, **options)
            else:
                result = {}
                tasks = ((signal_desc, a, b, q_cart, n_atoms, options) for a, b in ranges)
                for start, stop, spectra in _completed(executor.pool, _finish_signals, tasks):
                    for name, values in spectra.items():
                        if name not in result:
                            result[name] = np.empty((len(q_cart), block_size), dtype=values.dtype)
                        result[name][start:stop] = values
                    if advance is not None:
                        advance(stop - start)
            yield result
