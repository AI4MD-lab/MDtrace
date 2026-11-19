Input parameters
================

An input file contains ``name = value`` entries. Blank lines and text after
``#`` are ignored. Run ``mdtrace input.in``; without an argument, only
``input.in`` is selected. The default method is ``dsf``.

Common parameters
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Meaning
   * - ``action``
     - ``thinking``
     - thinking, compute, or plot. Thinking checks settings/source changes before reusing DSF output.
   * - ``method``
     - ``dsf``
     - DSF is the default. EELS remains planned; use pySED for SED.
   * - ``backend``
     - ``numpy``
     - Serial NumPy DSF. The SED CuPy backend has been removed.
   * - ``trajectory_file``
     - ``dump.xyz``
     - XYZ, LAMMPS dump or NetCDF. Relative paths resolve against the control file.
   * - ``out_files_name``
     - ``mdtrace``
     - Result prefix; relative paths resolve against the working directory.
   * - ``lammps_unit``
     - ``metal``
     - metal or real velocity convention; the reader converts velocities to m/s.
   * - ``trajectory_read_mode``
     - ``cache``
     - DSF requires cache. Native NetCDF is read directly; text is converted.
   * - ``trajectory_prefetch``
     - ``1``
     - Retained for input compatibility; DSF does not use the prefetch helper.
   * - ``netcdf_compression_level``
     - ``1``
     - Integer zlib level 0 through 9. Zero disables text-cache compression.
   * - ``netcdf_batch_size``
     - ``64``
     - Positive frame count per conversion batch; independent of statistical blocks.
   * - ``time_step``
     - ``0``
     - MD integration step in fs; positive for computation.
   * - ``output_data_stride``
     - ``0``
     - MD steps between saved frames; positive for computation.
   * - ``num_blocks``
     - ``5``
     - Equal non-overlapping DSF blocks; trailing frames are cropped with a warning.
   * - ``max_cores``
     - ``4``
     - Positive compatibility setting; DSF currently runs serially.
   * - ``prim_unitcell``
     - ``None``
     - Nine row-vector components in Angstrom; needed for fractional Q points and paths.

DSF parameters
--------------

See :doc:`dsf_workflow` for all DSF sampling, weighting, normalization, current,
partial-spectrum and plotting options. ``if_show_figures = 0`` saves and closes
plots; set it to ``1`` for interactive display. Boolean inputs accept 0 or 1.
No basis file, equilibrium-site map or phonon eigenvector is required.

The saved-frame interval is ``time_step * output_data_stride`` in fs. A single
statistical block cannot estimate SEM. Conversion batch size and compression
affect trajectory I/O, not the DSF normalization or block definitions.

The direct text and prefetch helpers remain in the Python I/O layer for
compatibility. They are not used by the current DSF workflow.
