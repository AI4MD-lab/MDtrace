Quick start
===========

Run ``mdtrace input.in`` or ``mdtrace`` to read ``input.in`` in the current
working directory. ``mdtrace -h`` displays supported actions. The default method
is ``dsf``. MDtrace no longer automatically opens ``input_SED.in``; use pySED
for SED calculations.

For a number-density spectrum at a periodic-box wave vector, create an input
such as the following. The example assumes a fixed 10-Angstrom cubic box and
positions saved every 10 MD steps at a 1-fs integration timestep. Replace these
values with those of your actual trajectory.

.. code-block:: ini

   method = dsf
   action = compute
   experiment = number
   trajectory_file = dump.xyz
   trajectory_read_mode = cache
   time_step = 1
   output_data_stride = 10
   num_blocks = 2
   out_files_name = sample
   dsf_q_coordinates = cartesian
   dsf_qpoints = 0.6283185307179586 0 0

Text input is converted to a reusable ``dump.xyz.mdtrace.nc`` beside the control
file. Native NetCDF is read directly. Trajectory paths are relative to the
control file; relative output prefixes are relative to the working directory.

``compute`` writes the numerical DSF outputs, including ``sample.dsf.npz``.
Change ``action`` to ``plot`` to draw the saved spectrum, or use ``thinking``
to compute missing/changed results and then plot. Plotting needs the archive,
not the original trajectory. ``action = fit`` is not supported.

Set ``dsf_currents = 1`` only when the same trajectory also contains velocities.
Use ``experiment = neutron`` with ``atom_types`` for neutron weights. Crystal
paths, amorphous shells, element pairs and QENS post-processing are described in
:doc:`dsf_workflow`. Worked control files are in ``examples/dsf/``.
