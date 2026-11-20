Common parameters
=================

Common parameters control the MDtrace workflow, trajectory input, time
sampling, data reading, performance, and general output. They are not tied to
one spectral observable. Every parameter name below opens its complete
reference page.

.. list-table::
   :class: parameter-index
   :header-rows: 1
   :widths: 27 23 16 44

   * - Parameter
     - Group
     - Default
     - Purpose
   * - :doc:`action <action>`
     - :doc:`Control <common_control>`
     - ``thinking``
     - Select automatic workflow, compute, plot, or fit.
   * - :doc:`method <method>`
     - :doc:`Control <common_control>`
     - ``sed``
     - Select the analysis method.
   * - :doc:`device <device>`
     - :doc:`Control <common_control>`
     - ``cpu``
     - Select NumPy or optional CuPy.
   * - :doc:`trajectory_file <trajectory_file>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``dump.xyz``
     - Name the trajectory input.
   * - :doc:`lammps_unit <lammps_unit>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``metal``
     - Select the LAMMPS velocity units.
   * - :doc:`num_atoms <num_atoms>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``0``
     - Set the trajectory atom count.
   * - :doc:`total_num_steps <total_num_steps>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``0``
     - Set the requested production length.
   * - :doc:`time_step <time_step>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``0.0``
     - Set the MD time step in fs.
   * - :doc:`output_data_stride <output_data_stride>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``0``
     - Set the saved-frame interval in MD steps.
   * - :doc:`num_blocks <num_blocks>`
     - :doc:`Trajectory and sampling <common_trajectory>`
     - ``5``
     - Set the number of averaging blocks.
   * - :doc:`trajectory_read_mode <trajectory_read_mode>`
     - :doc:`I/O and performance <common_io_performance>`
     - ``cache``
     - Choose cached or direct text reading.
   * - :doc:`trajectory_prefetch <trajectory_prefetch>`
     - :doc:`I/O and performance <common_io_performance>`
     - ``1``
     - Prepare one trajectory block ahead.
   * - :doc:`netcdf_batch_size <netcdf_batch_size>`
     - :doc:`I/O and performance <common_io_performance>`
     - ``64``
     - Set text parsing batch size.
   * - :doc:`netcdf_compression_level <netcdf_compression_level>`
     - :doc:`I/O and performance <common_io_performance>`
     - ``1``
     - Set converted-cache compression.
   * - :doc:`max_cores <max_cores>`
     - :doc:`I/O and performance <common_io_performance>`
     - ``4``
     - Limit NumPy worker processes.
   * - :doc:`out_files_name <out_files_name>`
     - :doc:`Output and display <common_output>`
     - ``mdtrace``
     - Set the output prefix.
   * - :doc:`if_show_figures <if_show_figures>`
     - :doc:`Output and display <common_output>`
     - ``0``
     - Show saved figures interactively.

:doc:`Back to the complete parameter index <../input_parameters>`

.. toctree::
   :hidden:
   :maxdepth: 2

   common_control
   common_trajectory
   common_io_performance
   common_output
