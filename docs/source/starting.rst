Quick start
===========

For the first calculation, choose one of two Q-sampling modes:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Use it for
     - What is plotted along the Q axis
   * - ``path``
     - Crystal vibrations along selected directions
     - Distance along a reciprocal-space path; directions remain distinct
   * - ``spherical``
     - Isotropic amorphous materials and liquids
     - Q magnitude; spectra are averaged over directions in each shell

Both examples calculate density, longitudinal-current (CL) and transverse-current
(CT) spectra. CT is the sum of the two transverse polarizations in three
dimensions. The trajectory must contain positions and velocities at the same
uniformly spaced times, with a fixed, fully periodic simulation cell. If only
positions are available, set ``dsf_currents = 0`` to calculate density alone.

Prepare ``input.in``
--------------------

Copy **one** of the complete templates below to ``input.in`` in a new calculation
directory. Put your trajectory in that directory and edit its filename,
``time_step`` and ``output_data_stride`` to match how it was generated. For a
crystal, also replace the example primitive lattice and reciprocal path. These
are templates for your own trajectory, not bundled production datasets.

* :download:`Crystal path input.in template <../../examples/dsf/crystal.in>`:
  see :ref:`dsf-student-path` for the complete input and coordinate explanation.
* :download:`Isotropic spherical input.in template <../../examples/dsf/amorphous.in>`:
  see :ref:`dsf-student-spherical` for the complete input and sampling explanation.

The repository keeps these files as ``examples/dsf/crystal.in`` and
``examples/dsf/amorphous.in`` so that the two templates can live together.
Rename your chosen copy to ``input.in``. Each template is complete; do not
combine the two. Both explicitly select their Q mode, so no implicit sampling
default is needed.

Run and read the results
------------------------

From your calculation directory, run:

.. code-block:: text

   mdtrace input.in

``mdtrace`` without arguments also reads ``input.in`` in the current directory.
The examples use ``action = thinking``: compute missing or changed results,
then plot directly from the saved text. ``action = compute`` always recomputes;
``action = plot`` only plots existing text. Plotting does not need the original
trajectory. ``action = fit`` is not supported.

For these examples, the numerical outputs are ``.dsf``, ``.dsf_THz``,
``.dsf_Qaxis``, ``.dsf_CL``, ``.dsf_CT`` and ``.dsf_Ctotal``, prefixed by ``out_files_name``.
The density figure is ``<prefix>.dsf.png``. The three current figures use
``.dsf_Ctotal.png``, ``.dsf_CL.png`` and ``.dsf_CT.png`` with the same prefix.
Spectrum tables have frequency rows and Q
columns. The workflow uses no NPZ result archive.

Text input is converted to a reusable ``<trajectory-name>.mdtrace.nc`` cache
beside ``input.in``; this is a trajectory cache, not a spectrum result. Native
NetCDF is read directly. Relative trajectory paths resolve against ``input.in``;
relative output prefixes resolve against the working directory. Using a
separate calculation directory and a distinct output prefix keeps each run
separate from existing results.

Directional and advanced calculations
-------------------------------------

There is no separate ``anisotropic`` mode. ``path`` retains directional
information for crystals. ``spherical`` averages directions and therefore
cannot show their differences. For anisotropic amorphous samples, advanced
users can retain specified directions with the underlying ``explicit`` input;
it is outside this introductory workflow. See :ref:`dsf-anisotropy`.

Neutron weighting, self spectra, species pairs and QENS post-processing are
described in :doc:`dsf_workflow`. SED calculations are provided by pySED.
