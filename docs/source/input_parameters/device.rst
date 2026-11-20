device
~~~~~~

**Syntax**

.. code-block:: text

   device = cpu
   # device = gpu

**Meaning**
   Selects the device for DSF calculation. ``cpu`` uses NumPy with one process when
   ``max_cores = 1`` and permits persistent Q-point workers when
   ``max_cores > 1``, subject to CPU and memory limits.
   ``gpu`` uses CuPy on one CUDA GPU with float64/complex128 precision. Frame and Q
   batches are selected automatically from available memory. Density/current
   streaming budgets up to 90% of available GPU memory and retries a smaller
   Q batch if allocation fails. Time signals stay on the GPU when they fit,
   otherwise in host RAM. GPU FFTs use complete time blocks in both cases.
   Self, neutron and partial spectra also
   use the GPU, with Q batching; their full atom-wise block must fit device memory.

**Allowed values**
   ``cpu`` or ``gpu``. GPU calculations require CuPy and a usable CUDA device.

**Default**
   ``cpu``.

``max_cores`` controls CPU workers. Use ``action = compute`` to time the same
calculation on another device.

:doc:`Back to Parameter Index <../input_parameters>`

Related parameters
------------------

- :doc:`max_cores <max_cores>`
- :doc:`trajectory_prefetch <trajectory_prefetch>`
