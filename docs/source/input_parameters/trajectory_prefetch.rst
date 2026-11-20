trajectory_prefetch
~~~~~~~~~~~~~~~~~~~

**Syntax**

.. code-block:: text

   trajectory_prefetch = 1

**Meaning**
   Enable (1) or disable (0) one-batch prefetch for streamed density/current
   DSF. One background thread uses the pySED reader to prepare the next small
   frame batch while the Q workers compute the current batch. Native NetCDF
   and converted text caches use the same path. No extra trajectory file is made.

   Internal frame batches do not change ``num_blocks`` or the time FFT length.
   Their size is automatically bounded by memory. The complete statistical
   block is assembled as density/current signals before subtracting the mean,
   applying the window and transforming in time.

**Default**
   ``1``.

**Scope and memory**
   Self spectra, nonzero neutron incoherent weights and partial spectra retain
   the full-block calculation, without prefetch. Currently
   ``dsf_output_components``, ``dsf_save_q_resolved`` and ``dsf_partials`` request
   atom-wise spectra and therefore select this path.

   Prefetch keeps at most one upcoming raw batch. Memory planning also includes
   shared input buffers, unit-conversion temporaries, complete time signals and
   spectrum outputs. Unknown available memory disables prefetch; limited RAM
   reduces batch size and may disable it. Unavailable shared storage uses the
   streamed serial path. Actual batch size and prefetch status are printed.

   NetCDF access is serialized: the metadata reader closes before the batch
   reader starts, and compute workers never open the trajectory. On early exit
   an active read finishes before its source and shared buffers are closed.

   Speedup depends on the balance of reading and computation. Compare complete
   wall time with prefetch enabled and disabled.

:doc:`Back to Parameter Index <../input_parameters>`
