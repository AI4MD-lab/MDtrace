I/O and performance
===================

These parameters control how trajectories reach the DSF calculation and how
much CPU parallelism or NetCDF compression is used.

.. list-table::
   :class: parameter-index
   :header-rows: 1
   :widths: 30 17 53

   * - Parameter
     - Default
     - Purpose
   * - :doc:`trajectory_read_mode <trajectory_read_mode>`
     - ``cache``
     - Convert text once to reusable NetCDF; native NetCDF is read directly.
   * - :doc:`trajectory_prefetch <trajectory_prefetch>`
     - ``1``
     - Prepare at most one next small batch for streamed density/current DSF.
   * - :doc:`netcdf_batch_size <netcdf_batch_size>`
     - ``64``
     - Set the number of text frames parsed per batch.
   * - :doc:`netcdf_compression_level <netcdf_compression_level>`
     - ``1``
     - Set zlib compression for newly converted NetCDF caches.
   * - :doc:`max_cores <max_cores>`
     - ``4``
     - Limit persistent NumPy worker processes.

Typical cache mode:

.. code-block:: text

   trajectory_read_mode       = cache
   trajectory_prefetch        = 1
   netcdf_compression_level   = 1
   max_cores                  = 8

:doc:`Back to Common parameters <common>`

.. toctree::
   :hidden:

   trajectory_read_mode
   trajectory_prefetch
   netcdf_batch_size
   netcdf_compression_level
   max_cores
