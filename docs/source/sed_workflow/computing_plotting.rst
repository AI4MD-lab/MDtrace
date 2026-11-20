Computing and plotting SED
==========================

Computation
-----------

After preparing the trajectory, basis mapping, primitive cell, and Q path,
force numerical calculation with

.. code-block:: text

   action  = compute
   method  = sed
   device = cpu

The NumPy backend is serial when ``max_cores = 1`` and uses persistent worker
processes when ``max_cores > 1``. ``device = gpu`` uses one CUDA GPU for the
SED kernel; ``max_cores`` does not control that kernel. Plotting and spectral
fitting remain CPU operations.

Trajectory blocks are accumulated online and averaged before the positive
frequency spectrum is written. :doc:`../input_parameters/num_blocks` trades
frequency resolution for statistical averaging. The implementation and
normalization are described in :doc:`../theory`.

CuPy timing report
------------------

The CuPy timing table lists trajectory-block wait, upload, projection, FFT,
other GPU operations, and result download. ``Sum of the six rows above`` adds
those six entries. ``Other SED work`` immediately below it accounts for setup,
allocation, accumulation, averaging, and cleanup inside the complete SED
calculation. Their sum equals ``Total SED calculation time`` up to displayed
rounding.

Plotting
--------

Regenerate figures from existing numerical data with

.. code-block:: text

   action               = plot
   plot_cutoff_freq     = 25
   plot_interval        = 5
   plot_slice           = 1
   qpoint_slice_index   = 23
   if_show_figures      = 0

The dispersion heatmap uses
:math:`\ln[\Phi(\mathbf q,f)/(1\,\mathrm{eV\,THz^{-1}})]` for color, while a
single-Q slice plots the untransformed SED in ``eV/THz`` on a logarithmic
vertical axis. See :doc:`../sed_units` for the exact convention.

Use :doc:`../input_parameters/sed_output_plotting` for the complete clickable
plotting-parameter index and :doc:`output_files` for filenames.
