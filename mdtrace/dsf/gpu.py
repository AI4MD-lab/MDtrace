# Copyright 2026 Ting Liang and MDTRACE development team
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional CuPy DSF with bounded frame/Q batches and complete time FFTs."""

from collections import deque
from contextlib import closing

import numpy as np
from pySED.console import print_detail, print_section
from pySED.io.trajectory import prefetch_one

from mdtrace.console import frame_range
from mdtrace.dsf.parallel import _memory_estimate
from mdtrace.dsf.spectra import _spectral_scale, block_spectra
from mdtrace.dsf.streaming import _batch_plan, _trajectory_batches


def _stream_plan(n_atoms, block_size, n_q, currents, batch, budget):
    """Reserve full time signals when possible, then bound phase/FFT scratch."""
    channels = 4 if currents else 1
    signal_bytes = n_q * channels * block_size * 16
    resident = signal_bytes <= budget // 2
    while True:
        inputs = batch * n_atoms * 3 * 8 * (2 if currents else 1) * 2
        # The in-place phase uses 16 bytes per atom/frame. Allow another 16
        # for the real dot product and matrix-multiplication scratch.
        per_q = 32 * batch * n_atoms + 128 * channels * block_size
        available = budget - inputs - n_atoms * 16 - n_q * 24 - (signal_bytes if resident else 0)
        if available >= per_q:
            capacity = min(n_q, available // per_q)
            batches = (n_q + capacity - 1) // capacity
            return batch, (n_q + batches - 1) // batches, resident
        if batch > 1:
            batch = max(1, batch // 2)
        elif resident:
            resident = False
        else:
            raise MemoryError("Insufficient GPU memory for one DSF Q point; use device = cpu")


def _q_batch_summary(n_q, q_batch):
    vectors = "Q vector" if n_q == 1 else "Q vectors"
    if n_q <= q_batch:
        return f"{n_q} {vectors} in 1 batch"
    full, remainder = divmod(n_q, q_batch)
    batches = full + bool(remainder)
    split = f"{full} × {q_batch}"
    if remainder:
        split += f" + {remainder}"
    return f"{n_q} {vectors} in {batches} batches ({split})"


def _is_device_oom(cp, error):
    return (isinstance(error, cp.cuda.memory.OutOfMemoryError)
            or (isinstance(error, cp.cuda.runtime.CUDARuntimeError) and error.status == 2)  # cudaErrorMemoryAllocation
            or (isinstance(error, cp.cuda.cublas.CUBLASError) and error.status == 3)  # CUBLAS_STATUS_ALLOC_FAILED
            or (isinstance(error, cp.cuda.cufft.CuFFTError) and error.result == 2))  # CUFFT_ALLOC_FAILED


def _run_q_batches(cp, pool, n_q, q_batch, calculate):
    """Advance only after the entire Q slice has been written successfully."""
    start = 0
    while start < n_q:
        stop = min(n_q, start + q_batch)
        try:
            calculate(start, stop)
        except (MemoryError, RuntimeError) as error:
            if not _is_device_oom(cp, error):
                raise
            if stop - start == 1:
                raise MemoryError("Insufficient GPU memory for one DSF Q vector; use device = cpu") from error
        else:
            start = stop
            yield stop, q_batch
            continue
        # Drop the exception traceback (and temporary GPU arrays) before retrying.
        cp.cuda.get_current_stream().synchronize()
        pool.free_all_blocks()
        q_batch = (stop - start) // 2


def _fft_spectra(cp, signals, q_cart, n_atoms, *, dt_ps,
                 normalization="density", subtract_mean=False):
    """Transform a Q batch; time is contiguous and retains the full block."""
    if subtract_mean:
        signals = signals - signals.mean(axis=-1, keepdims=True)
    transformed = cp.fft.fftshift(cp.fft.fft(signals, axis=-1), axes=-1)
    scale = _spectral_scale(signals.shape[-1], n_atoms, dt_ps, normalization)
    density = abs(transformed[:, 0])**2 * scale
    result = {"coherent_sqw": density, "incoherent_sqw": cp.zeros_like(density), "sqw": density}
    if signals.shape[1] == 4:
        norms = np.linalg.norm(q_cart, axis=1)
        directions = np.divide(q_cart, norms[:, None], out=np.zeros_like(q_cart), where=norms[:, None] != 0)
        directions = cp.asarray(directions)
        current = transformed[:, 1:].transpose(0, 2, 1)
        along = cp.einsum("qtd,qd->qt", current, directions)
        across = current - along[:, :, None] * directions[:, None, :]
        result["longitudinal_current"] = abs(along)**2 * scale
        result["transverse_current"] = (abs(across)**2).sum(axis=-1) * scale
        result["longitudinal_current"][norms == 0] = cp.nan
        result["transverse_current"][norms == 0] = cp.nan
    return result


class GPUBackend:
    """One GPU; NumPy output and the same normalization as the CPU backend."""

    def __enter__(self):
        # Reuse pySED's lazy import and CUDA-device checks.
        from pySED.backends.gpu import _load_cupy

        self.cp = _load_cupy()
        free, total = self.cp.cuda.runtime.memGetInfo()
        properties = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        name = properties["name"]
        print_section("GPU resources", "🚀")
        print_detail("Device", name.decode() if isinstance(name, bytes) else name)
        print_detail("Total GPU memory", f"{total / 1024**3:.2f} GiB")
        print_detail("Available GPU memory", f"{free / 1024**3:.2f} GiB")
        return self

    def __exit__(self, *exc):
        self.cp.cuda.get_current_stream().synchronize()
        self.cp.get_default_memory_pool().free_all_blocks()

    def _budget(self, *, reuse_cached=True):
        pool = getattr(self.cp.cuda.get_allocator(), "__self__", None)
        available = self.cp.cuda.runtime.memGetInfo()[0]
        if isinstance(pool, self.cp.cuda.MemoryPool):
            reserved = pool.total_bytes()
            if reuse_cached:
                available += pool.free_bytes()
                reserved = pool.used_bytes()
            if pool.get_limit():
                available = min(available, max(0, pool.get_limit() - reserved))
        return int(available * 0.90)

    def compute(self, positions, q_cart, coherent, incoherent, *, velocities=None, progress=None, **options):
        """Reuse the estimator for self, neutron and complex partial spectra."""
        cp = self.cp
        shared, outputs, scratch = _memory_estimate(positions, velocities, len(q_cart), incoherent, options)
        per_q = max(1, outputs // len(q_cart))
        batch = min(len(q_cart), (self._budget() - shared - scratch) // per_q)
        if batch < 1:
            raise MemoryError("Atom-wise DSF block exceeds available GPU memory; use device = cpu")
        gpu_positions = cp.asarray(positions, dtype=cp.float64)
        gpu_velocities = cp.asarray(velocities, dtype=cp.float64) if velocities is not None else None
        gpu_coherent, gpu_incoherent = cp.asarray(coherent), cp.asarray(incoherent)
        result = {}
        if progress is not None:
            progress(0)
        for start in range(0, len(q_cart), batch):
            stop = min(len(q_cart), start + batch)
            spectra = block_spectra(gpu_positions, q_cart[start:stop], gpu_coherent, gpu_incoherent,
                                    velocities=gpu_velocities, xp=cp, **options)
            for name, values in spectra.items():
                values = cp.asnumpy(values)
                if name not in result:
                    result[name] = np.empty((len(q_cart), *values.shape[1:]), dtype=values.dtype)
                result[name][start:stop] = values
            del spectra
            if progress is not None:
                progress(stop / len(q_cart))
        return result

    def stream_blocks(self, path, block_size, num_blocks, n_atoms, q_cart, coherent, **options):
        """Bound streaming allocations without modifying the caller's CuPy pool limit."""
        cp = self.cp
        allocator = cp.cuda.get_allocator()
        default_pool = cp.get_default_memory_pool()
        uses_default = getattr(allocator, "__self__", None) is default_pool
        if uses_default:
            # Use one cache for DSF arrays, releasing unused default-pool blocks first.
            default_pool.free_all_blocks()
        budget = self._budget(reuse_cached=not uses_default)
        if budget < 1:  # A zero CuPy pool limit means unlimited.
            raise MemoryError("No GPU memory available for DSF; use device = cpu")
        pool = cp.cuda.MemoryPool() if uses_default else cp.cuda.MemoryPool(allocator)
        pool.set_limit(size=budget)
        blocks = self._stream_blocks(pool, path, block_size, num_blocks, n_atoms, q_cart, coherent, **options)
        try:
            while True:
                with cp.cuda.using_allocator(pool.malloc):
                    result = next(blocks, None)
                if result is None:
                    break
                yield result
        finally:
            blocks.close()
            cp.cuda.get_current_stream().synchronize()
            pool.free_all_blocks()

    def _stream_blocks(self, pool, path, block_size, num_blocks, n_atoms, q_cart, coherent,
                       *, currents=False, prefetch=True, atom_order=None, progress=None, batch_status=None,
                       trajectory_frames=None, **options):
        """Upload each frame batch once, reusing it for all Q batches."""
        cp = self.cp
        batch, _, prefetch = _batch_plan(n_atoms, block_size, len(q_cart), currents, 1, prefetch)
        batch, q_batch, resident = _stream_plan(n_atoms, block_size, len(q_cart), currents, batch, pool.get_limit())
        channels = 4 if currents else 1
        shape = (len(q_cart), channels, block_size)
        if resident:
            try:
                signals = cp.empty(shape, dtype=complex)
            except (MemoryError, RuntimeError) as error:
                if not _is_device_oom(cp, error):
                    raise
                resident = False
        if not resident:
            signals = np.empty(shape, dtype=complex)
        gpu_q = cp.asarray(q_cart)
        gpu_weights = cp.asarray(coherent, dtype=complex)
        print_section("Trajectory streaming", "🚀")
        print_detail("Batch size", f"{batch} frames")
        print_detail("GPU Q batch", _q_batch_summary(len(q_cart), q_batch))
        print_detail("Prefetch", "1 (one batch ahead)" if prefetch else "0")
        n_q = len(q_cart)
        if trajectory_frames is None:
            trajectory_frames = block_size * num_blocks
        stream = cp.cuda.get_current_stream()
        reported_batch = q_batch
        source = _trajectory_batches(path, block_size, num_blocks, batch, currents, atom_order)
        with closing(source), prefetch_one(source, enabled=prefetch) as batches:
            for block in range(num_blocks):
                block_start = block * block_size
                if progress is not None:
                    progress(block)
                for offset in range(0, block_size, batch):
                    positions, velocities = next(batches)
                    size = min(batch, block_size - offset)
                    if positions.shape != (size, n_atoms, 3) or not np.isfinite(positions).all():
                        raise ValueError("trajectory positions must be finite and match the requested batch")
                    if currents and (velocities.shape != positions.shape or not np.isfinite(velocities).all()):
                        raise ValueError("velocities must be finite and match positions (frame, atom, 3)")
                    gpu_positions = cp.asarray(positions, dtype=cp.float64)
                    gpu_velocities = cp.asarray(velocities, dtype=cp.float64) if currents else None
                    del positions, velocities
                    if batch_status is not None:
                        first_frame = block_start + offset + 1
                        stage = frame_range(first_frame, first_frame + size - 1, trajectory_frames)
                        completed = block + min(offset / block_size, .99)
                        batch_status(completed, stage, 0, n_q)
                        pending = deque()

                    def fill_signals(start, stop):
                        phase = cp.empty((size, stop - start, n_atoms), dtype=complex)
                        phase.real = 0
                        phase.imag = gpu_q[start:stop] @ gpu_positions.transpose(0, 2, 1)
                        cp.exp(phase, out=phase)
                        density = (phase @ gpu_weights).T
                        signals[start:stop, 0, offset:offset + size] = density if resident else cp.asnumpy(density)
                        if currents:
                            current = (phase @ gpu_velocities).transpose(1, 2, 0)
                            signals[start:stop, 1:, offset:offset + size] = current if resident else cp.asnumpy(current)

                    for stop, q_batch in _run_q_batches(cp, pool, n_q, q_batch, fill_signals):
                        if batch_status is not None:
                            # Mark finished work without blocking submission of the next Q batch.
                            event = cp.cuda.Event(disable_timing=True)
                            event.record(stream)
                            pending.append((event, stop))
                            while pending and pending[0][0].done:
                                _, done = pending.popleft()
                                batch_status(completed, stage, done, n_q)
                    del gpu_positions, gpu_velocities
                    if batch_status is not None:
                        for event, done in pending:
                            event.synchronize()
                            batch_status(completed, stage, done, n_q)
                    stream.synchronize()
                    if progress is not None:
                        progress(block + min((offset + size) / block_size, .99))
                result = {}
                if batch_status is not None:
                    fft_stage = "FFT · " + frame_range(block_start + 1, block_start + block_size, trajectory_frames)
                    batch_status(block + .99, fft_stage, 0, n_q)

                def collect_spectra(start, stop):
                    spectra = _fft_spectra(cp, cp.asarray(signals[start:stop]), q_cart[start:stop], n_atoms, **options)
                    for name, values in spectra.items():
                        values = cp.asnumpy(values)
                        if name not in result:
                            result[name] = np.empty((len(q_cart), block_size), dtype=values.dtype)
                        result[name][start:stop] = values

                for stop, q_batch in _run_q_batches(cp, pool, n_q, q_batch, collect_spectra):
                    if batch_status is not None:
                        batch_status(block + .99, fft_stage, stop, n_q)
                yield result
                if q_batch != reported_batch:
                    print_detail("GPU Q batch after retry", _q_batch_summary(n_q, q_batch))
                    reported_batch = q_batch
