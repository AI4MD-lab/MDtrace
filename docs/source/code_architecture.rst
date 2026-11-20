Code architecture and reading guide
============================================

MDtrace currently calculates DSF and particle-current spectra. The package root
owns the command line, input configuration and workflow. The ``io`` package
delegates trajectory reading to pySED; ``dsf`` owns the physics and its saved
results. SED is provided separately by pySED. MDtrace requires
``pysed-phonon>=3.0`` and imports the readers from its I/O submodules.
With pySED 3.0, GPUMD text conversion includes an extra header scan to retain
per-frame cells and check property layouts. Native NetCDF and validated cache
reuse need no text scan. Newer pySED versions perform these checks during parsing.

Module responsibilities
-----------------------

.. code-block:: text

   mdtrace/
     main.py              Command line, help and user-facing errors
     parser.py            Input syntax, defaults and relative input paths
     parameters.py        Parameter definitions and semantic validation
     pipeline.py          Prepare trajectory; choose compute or plot
     io/
       __init__.py        Public pySED reader and canonical field imports
       prepare.py         MDtrace cache names and checked conversion policy
     dsf/
       dsf.py             Trajectory checks, atom alignment, block calculation
       spectra.py         FFT periodograms, normalization and block statistics
       parallel.py        Persistent Q workers, shared blocks and memory limits
       streaming.py       Bounded frame batches, shared signals and one-batch prefetch
       partials.py        Species pairs and reconstruction of weighted spectra
       scattering.py      Coherent/incoherent neutron weights and isotope aliases
       neutron_scattering_lengths.py
                          Reviewed scattering data
       qpoints.py         Explicit/path/spherical sampling and shell diagnostics
       commensurate.py    Exact reciprocal-path geometry
       results.py         Result metadata, text checks, loading and writing
       postprocess.py     Reweighting, shell averaging and resolution convolution
       plot.py            Plot saved spectra
     eels/                Placeholder only; no EELS implementation

Follow a calculation
--------------------

1. ``main.main`` reads input through ``parser.read_input``. Defaults and input
   validation come from ``parameters``.
2. ``pipeline.step_prepare_trajectory`` selects a native NetCDF trajectory or
   converts text through ``io.prepare.prepare_trajectory``, which delegates to
   ``pySED.io.prepare.prepare_trajectory`` (with header validation where
   supported). The optimized pySED parser and chunked/compressed writer remain
   the conversion implementation; MDtrace supplies the 3.0 header checks.
3. ``dsf.compute_dsf`` reads the trajectory, checks sampling/cells/atom IDs,
   builds allowed Q vectors and prepares scattering weights.
4. ``parallel.DSFExecutor`` distributes Q ranges within a time block, sharing
   positions/velocities and reusing the process pool across blocks. It reuses
   pySED's CPU-memory query and Q-batching helpers; its memory estimate is DSF-specific.
   Ordinary density/current spectra use ``streaming.stream_blocks`` to fill
   complete time signals from small frame batches, with pySED's ``prefetch_one``.
   Atom-wise spectra retain ``spectra.block_spectra``. Both paths use the same
   time-transform, normalization and current-projection helpers in ``spectra``. ``qpoints.average_q_groups`` averages directions;
   ``spectra.mean_and_sem`` combines block estimates.
   With ``device = gpu``, ``gpu.GPUBackend`` instead batches density/current
   projection and complete time FFTs on one GPU. Atom-wise spectra reuse
   ``block_spectra`` with the CuPy array module; partial reconstruction shares
   the same formulas. Only completed spectra return to NumPy for averaging/output,
   unless limited device memory requires host storage of the reduced time signals.
5. The calculator returns arrays and metadata. The pipeline separately calls
   ``results.save_dsf`` to write spectrum/axis text files with metadata in the Q-axis header.

Numerical kernels do not import the command line, trajectory readers or plotter.
The calculation does not write analysis results itself. Output filenames and
text metadata belong to ``results``. Settings/source checks for ``thinking``
also live there; the pipeline only decides which step to execute.

Read and post-process results
-----------------------------

``results.load_spectra`` is the shared text reader for plotting and
post-processing. With ``for_plot=True`` it loads only displayed spectra and
the Q information needed to draw them; missing optional SEM, self-spectrum or
block files do not prevent plotting. Full loading retains post-processing data.
Neither use requires the original trajectory or imports the
trajectory calculator. Reweighting calls ``scattering.neutron_weights``;
statistical aggregation calls ``spectra.mean_and_sem``. The reviewed scattering
table is shared, without duplicating values.

Public entry points are:

.. code-block:: python

   from mdtrace.dsf import compute_dsf, save_dsf, load_spectra
   from mdtrace.dsf.postprocess import average_shells, reweight_scattering

   result = compute_dsf(params)
   save_dsf(result, params)
   saved = load_spectra(params.out_files_name)

The package initializer loads these functions on demand. Importing only a
numerical or result module does not load trajectory computation or Matplotlib.
The older ``dsf.dsf.save_dsf`` and ``dsf.postprocess.load_spectra`` imports remain
aliases of the same implementations.

Where to make a change
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Intended change
     - Starting point
   * - DSF formula, FFT normalization or block statistics
     - ``dsf/spectra.py`` and numerical reference tests
   * - Atom order, sampling interval or fixed-cell validation
     - ``dsf/dsf.py``
   * - Crystal paths or amorphous shell sampling
     - ``dsf/qpoints.py`` and ``dsf/commensurate.py``
   * - Result format, filenames or reuse checks
     - ``dsf/results.py``
   * - Neutron weights or isotope aliases
     - ``dsf/scattering.py``; preserve reviewed table provenance
   * - Trajectory parsing, units or cache validity
     - Public ``pySED.io``; MDtrace policy stays in ``io/prepare.py``
   * - User options or command-line stages
     - ``parameters.py``, ``parser.py`` and ``pipeline.py``

Keep one implementation per numerical operation. Add modules for distinct
responsibilities, not individual small helpers. Future trajectory analyses
should share ``io`` without depending on DSF-specific formulas or output files.
The DSF block loop uses ``NetCDFReader.read(field, frames)``. Do not use pySED's
SED block source: it intentionally omits positions. Native NetCDF reads stay
field-selective; text conversion uses bounded batches. GPUMD frame headers are
checked during conversion, with identical consecutive headers parsed only once.
DSF still owns atom alignment, fixed-cell/time checks and m/s to Angstrom/ps
conversion for particle currents. Result filenames and formats are unchanged.

Common parsing, precision, compression and cache tests belong to pySED. MDtrace
keeps integration tests in ``tests/io/test_pysed_integration.py`` and its DSF
numerical tests. The former duplicate reader and SED block implementations have
been removed.

Run ``python -B -m pytest -q`` from the repository root. Tests cover numerical
reference sums, invariants, atom alignment, both text formats, cache behavior,
CLI workflows, output round trips, and import boundaries. The dynasor comparison
requires that optional dependency. The ``-B`` flag avoids recreating bytecode
caches; normal Python execution may recreate caches for active modules.
