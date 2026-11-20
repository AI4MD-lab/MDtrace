max_cores
~~~~~~~~~

**Syntax**

.. code-block:: text

   max_cores = 4

**Meaning**
   Maximum number of DSF CPU worker processes. ``max_cores = 1`` runs serially;
   larger values distribute Q points across persistent processes, each using
   one numerical-library thread.
   This setting does not control ``device = gpu``, which uses one GPU.

**Default**
   ``4``.

**Notes**
   Ordinary density/current DSF shares a small frame batch read-only.
   Atom-wise self/incoherent/partial calculations share a complete time block.
   Each worker calculates Q points sequentially with its own temporary arrays.
   Actual process count is capped by available CPUs, Q points, RAM and shared
   memory capacity, and printed before calculation. Memory estimates include
   extra FFT arrays for self, incoherent and partial spectra. Unknown memory
   capacity or unavailable shared storage falls back to serial execution.
   Time blocks remain sequential; this setting does not change frequency
   resolution, normalization, Q sampling or output files.

:doc:`Back to Parameter Index <../input_parameters>`
