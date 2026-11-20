# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persistent Q-point workers sharing one trajectory block, as in pySED."""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from contextlib import ExitStack, contextmanager
from multiprocessing import get_context, shared_memory

import numpy as np
from threadpoolctl import threadpool_limits
from pySED.backends.common import available_cpu_memory, qpoint_batches
from pySED.console import print_detail, print_section

from mdtrace.dsf.spectra import block_spectra


def _initialize_worker():
    # Keep the controller alive for the worker's lifetime.
    global _thread_limit
    _thread_limit = threadpool_limits(limits=1)


def _compute_batch(shared, q_cart, coherent, incoherent, options):
    with ExitStack() as stack:
        arrays = []
        for name, shape in shared:
            segment = shared_memory.SharedMemory(name=name)
            stack.callback(segment.close)
            array = np.ndarray(shape, dtype=np.float64, buffer=segment.buf)
            array.setflags(write=False)
            arrays.append(array)
        return block_spectra(
            arrays[0], q_cart, coherent, incoherent,
            velocities=arrays[1] if len(arrays) > 1 else None, **options,
        )


@contextmanager
def _share(values):
    # Copy/convert directly into shared storage, without a float64 staging copy.
    segment = shared_memory.SharedMemory(create=True, size=values.size * 8)
    try:
        array = np.ndarray(values.shape, dtype=np.float64, buffer=segment.buf)
        np.copyto(array, values)
        yield segment.name, values.shape
    finally:
        segment.close()
        segment.unlink()


def _memory_estimate(positions, velocities, n_q, incoherent, options):
    """Conservative extra bytes: shared inputs, collected outputs, per worker.

    Existing parent arrays are already reflected in available RAM. Per-worker
    estimates include phase/FFT temporaries and interpreter/library overhead.
    They are safety estimates, not predicted peak RSS.
    """
    frames, atoms, _ = positions.shape
    shared = positions.size * 8 + (velocities.size * 8 if velocities is not None else 0)
    labels = options.get("species_labels")
    atom_fft = options.get("compute_self", True) or labels is not None or np.any(incoherent)
    per_element = 160 if labels is not None else (128 if atom_fft else 40)
    channels = 6
    if labels is not None:
        species = len(np.unique(labels))
        pairs = species * (species + 1) // 2
        channels += 6 * pairs + 2 * species
    # Allow parent output, worker results, and transport buffers together.
    outputs = 3 * n_q * frames * channels * 16
    worker = frames * atoms * per_element + 128 * 1024**2
    return shared, outputs, worker


def _worker_count(requested, n_q, shared, outputs, worker):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    count = min(requested, cpus, n_q)
    if count <= 1:
        return 1
    available = available_cpu_memory()
    if available is None:
        return 1
    # Linux shared memory has a separate filesystem limit (notably in WSL).
    if os.name == "posix":
        try:
            info = os.statvfs("/dev/shm")
            if shared > info.f_bavail * info.f_frsize // 2:
                return 1
        except OSError:
            return 1
    # Leave 20% of currently available RAM free, on top of conservative arrays.
    return max(1, min(count, (int(available * 0.8) - shared - outputs) // worker))


class DSFExecutor:
    """Reuse a spawn pool across time blocks; each worker has one BLAS thread."""

    def __init__(self, max_cores=1):
        if int(max_cores) != max_cores or max_cores < 1:
            raise ValueError("max_cores must be a positive integer")
        self.max_cores = int(max_cores)
        self.workers = None
        self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def _configure(self, workers):
        if workers == self.workers:
            return
        self.close()
        self.workers = workers
        if workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize_worker,
            )
        print_section("CPU resources", "🚀")
        print_detail("Core limit", f"{self.max_cores} (max_cores)")
        print_detail("CPU allocation", f"{workers} process(es) × 1 thread")
        if workers < self.max_cores:
            print_detail("Allocation limit", "Q count, available CPUs or memory")

    def compute(self, positions, q_cart, coherent, incoherent, *, velocities=None, progress=None, **options):
        shared_bytes, outputs, working = _memory_estimate(
            positions, velocities, len(q_cart), incoherent, options,
        )
        workers = _worker_count(self.max_cores, len(q_cart), shared_bytes, outputs, working)
        self._configure(workers)
        if workers == 1:
            if progress is not None:
                progress(0)
            with threadpool_limits(limits=1):
                return block_spectra(positions, q_cart, coherent, incoherent,
                                     velocities=velocities, progress=progress, **options)

        with ExitStack() as stack:
            try:
                shared = [stack.enter_context(_share(positions))]
                if velocities is not None:
                    shared.append(stack.enter_context(_share(velocities)))
            except (OSError, MemoryError) as error:
                stack.close()
                warnings.warn(f"Shared memory unavailable; using serial DSF: {error}", RuntimeWarning)
                self._configure(1)
                if progress is not None:
                    progress(0)
                with threadpool_limits(limits=1):
                    return block_spectra(positions, q_cart, coherent, incoherent,
                                         velocities=velocities, progress=progress, **options)
            futures = {}
            result = {}
            completed = 0
            if progress is not None:
                progress(0)
            try:
                for start, stop in qpoint_batches(len(q_cart), workers):
                    future = self.pool.submit(
                        _compute_batch, shared, q_cart[start:stop], coherent, incoherent, options,
                    )
                    futures[future] = (start, stop)
                for future in as_completed(futures):
                    start, stop = futures[future]
                    for name, values in future.result().items():
                        if name not in result:
                            result[name] = np.empty((len(q_cart), *values.shape[1:]), dtype=values.dtype)
                        result[name][start:stop] = values
                    completed += stop - start
                    if progress is not None:
                        progress(completed / len(q_cart))
            finally:
                # Workers must detach before the parent's shared segments disappear.
                for future in futures:
                    future.cancel()
                wait(futures)
            return result
