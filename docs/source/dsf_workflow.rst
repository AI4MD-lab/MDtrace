DSF for crystals and amorphous systems
======================================

The development checkout supports classical density dynamic structure factors
and longitudinal/transverse particle-current spectra. The same coordinate-based
FFT kernel applies to crystals and disordered systems; no equilibrium-site map,
``basis.in``, or phonon eigenvectors are used. This addition is separate from the
published 1.1.0 SED-only release. DSF is now the default method; SED analysis
is provided separately by pySED.

Student inputs: path and spherical
----------------------------------

Start with one of the two complete inputs below. Copy it to ``input.in`` in
its own calculation directory, supply your trajectory, and run
``mdtrace input.in`` there. The two examples use the same density/current
calculation; only the selection and averaging of Q vectors differ.
``explicit`` remains an advanced interface and is not needed for these lessons.

.. _dsf-student-path:

Crystal: path input.in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this for direction-dependent crystal spectra. Download
:download:`crystal.in <../../examples/dsf/crystal.in>` and save your working
copy as ``input.in``:

.. literalinclude:: ../../examples/dsf/crystal.in
   :language: ini

``prim_unitcell`` gives three primitive real-space lattice vectors as rows in
Å. The Si values above are an example, not a universal cell. The cell in
the trajectory must be an integer supercell of this primitive lattice, in the
same Cartesian orientation.

Each triple in ``dsf_q_path`` is an endpoint expressed in the **primitive
reciprocal basis**, not a Cartesian vector in Å⁻¹. For primitive cell
matrix A, the Cartesian vector is ``Q = 2*pi*q*inverse(A).T``. The example
connects G to X to L. Labels only name the endpoints; they do not define the
geometry. MDtrace selects the simulation-cell-compatible Q points along each
segment, rather than inventing an arbitrary number of interpolated points.
A larger supercell can provide more points along the same path.

.. _dsf-student-spherical:

Isotropic amorphous material: spherical input.in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this for spectra averaged over directions in an isotropic amorphous
material or liquid. Download
:download:`amorphous.in <../../examples/dsf/amorphous.in>` and save your working
copy as ``input.in``:

.. literalinclude:: ../../examples/dsf/amorphous.in
   :language: ini

No primitive lattice or ``basis.in`` is needed. Q vectors come from the
simulation box's reciprocal lattice. ``dsf_q_min`` and ``dsf_q_max`` select
magnitudes in Å⁻¹; ``dsf_q_bin_width`` sets the shell width in the same
unit. Spectra, not complex amplitudes, are averaged within each shell. The
plotted Q is the mean magnitude of the sampled vectors, not necessarily the
nominal bin center. Empty shells are omitted and Q=0 is excluded.

``dsf_max_points = 5000`` sets the total vector budget across all shells,
counting both members of each +/- pair. Use ``0`` for all available vectors.
Positive budgets must be at least 2, and odd budgets round down to an even
number. Sampling favors low Q and thins dense high-Q regions to balance the
radial sampling. ``dsf_seed`` makes the selection reproducible. Changing
``dsf_q_bin_width`` groups the same selected vectors into different shells.

The pySED-style calculation summary prints the frame interval, block duration,
frequency spacing, Nyquist frequency, selected/allowed Q counts, and shell
coverage. A small total budget can leave some populated shells unsampled;
the summary reports their number and the plot leaves gaps. Check sampled
versus available counts in ``.dsf_Qaxis`` and directions in the screen summary. Larger budgets can
add available vectors, while box size/shape determines which directions exist.
Check shell width, budget and box size for the intended material and Q range.

Existing inputs with ``dsf_max_points_per_shell`` retain their per-shell
sampling behavior. Supply one budget parameter per input file.

Shared parameters and outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Parameters used in both teaching inputs
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``method = dsf`` / ``action = thinking``
     - Compute missing or changed DSF results and then plot their saved text. ``compute`` always recomputes; ``plot`` only reads existing text.
   * - ``density_weighting = equal``
     - Unit atom weights for density. This is not a neutron-weighted intensity.
   * - ``trajectory_file`` / ``trajectory_read_mode = cache``
     - Your trajectory; text is converted once to a checked reusable NetCDF cache. Native NetCDF is read directly.
   * - ``out_files_name``
     - Prefix for all result files. Use a distinct prefix or directory for each calculation.
   * - ``time_step`` / ``output_data_stride``
     - MD step in fs / MD steps between saved frames. Their product is the saved-frame interval: here 10 fs = 0.01 ps.
   * - ``num_blocks = 4``
     - Four equal non-overlapping time blocks. More blocks at fixed trajectory length give shorter blocks and coarser frequency resolution. Trailing frames may be cropped.
   * - ``dsf_currents = 1``
     - Add CL and CT to density; requires velocities. CT sums both transverse polarizations. Set 0 for density only.
   * - ``current_weighting = equal``
     - Particle currents with equal atom weights. Density weighting does not change the current weights.
   * - ``dsf_subtract_mean = 1``
     - Remove each amplitude's time mean within a block. Suitable here for vibrations; use 0 when retaining elastic/quasielastic contributions is intended.
   * - ``dsf_plot_frequency_max = 20``
     - Display up to 20 THz; saved spectra retain the full signed frequency range.

These inputs write six numerical text files: ``.dsf`` (density), ``.dsf_CL``
(longitudinal), ``.dsf_CT`` (transverse), ``.dsf_Ctotal`` (total current), ``.dsf_THz`` (frequency) and
``.dsf_Qaxis`` (Q coordinates and metadata). Plotting reads those files directly
and saves a separate PNG for each selected spectrum. The trajectory cache is separate from these results.
Standard frequency-density normalization is automatic: these equal-weight
examples give density spectra in 1/THz and currents in (Å/ps)^2/THz.

With a saved-frame interval of 0.01 ps, the Nyquist limit is 50 THz. Frequency
spacing also depends on the number of frames in each block; a 20-THz plot limit
does not improve resolution. At Q=0, CL and CT are undefined and saved as NaN.

.. _dsf-anisotropy:

What about anisotropic materials?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no separate ``dsf_q_mode = anisotropic``. Anisotropy means that spectra
can differ for vectors with the same magnitude but different directions.
``path`` keeps that direction information for crystals; compare paths along
the directions of interest using the appropriate primitive reciprocal basis.

``spherical`` averages these directions together. It can provide an angular
average if that is the intended observable, but cannot resolve anisotropy.
An elongated simulation box is also a sampling-geometry issue, not by itself
evidence of physical anisotropy.

For an anisotropic amorphous sample without a crystal primitive lattice,
advanced users can select individual Cartesian Q vectors through the retained
``explicit`` interface and compare directions without shell averaging. No
cone/sector averaging mode or separate transverse-polarization output is
implemented. All modes still require a fixed, fully periodic cell; this is not
a dedicated nonperiodic-film calculation.

Scope and observables
---------------------

Use a fixed, fully periodic simulation cell and uniformly sampled coordinates.
Particle currents additionally require velocities from the same frames and atom
ordering. Computation uses NumPy or optional CuPy and NetCDF input; text input is
converted through ``trajectory_read_mode = cache``. Variable-cell trajectories,
X-ray form-factor databases and automatic peak/lifetime fitting are not
included. The independent post-processing module supports neutron reweighting,
shell averaging, Gaussian/Lorentzian energy-resolution convolution and per-meV
spectral densities for classical neutron/QENS analysis.

``density_weighting = equal`` (the default) computes the number-density spectrum with unit weights.
Its ``.dsf`` equals its coherent collective spectrum; the neutron-incoherent
output is zero. The separate ``.dsf_self`` is the unweighted single-particle
spectrum and is already part of the collective spectrum, so it must not be added
to it again. ``density_weighting = neutron`` uses the existing Sears/NIST bound neutron
table without changing its entries. Missing scattering lengths raise an error.

``current_weighting = equal`` is the default and currently the only current
weighting: particle currents retain unit atom weights, including when density
uses neutron weights. Mass and X-ray weighting are not implemented and are
rejected. The old input ``experiment = number`` maps to
``density_weighting = equal``; ``experiment = neutron`` maps to
``density_weighting = neutron``. Do not put both parameter names in one input.
Existing text results remain readable, and changing only this spelling does
not force ``action = thinking`` to recompute matching results.

For the coherent part the instantaneous amplitude is

.. math::

   \rho_b(\mathbf Q,t)=\sum_j b_j e^{i\mathbf Q\cdot\mathbf r_j(t)}.

Complex amplitudes are summed before taking their squared modulus. Incoherent
neutron spectra sum the individual phase power spectra weighted by
:math:`|b_{j,\mathrm{inc}}|^2`. Natural-element entries include the source's
isotopic average; explicit isotope labels use their own entries. ``D`` and ``T``
are aliases for ``2H`` and ``3H``. Supply ``atom_types`` in ascending trajectory
type-ID order. GPUMD XYZ conversion assigns IDs by first appearance of each
species. Distinct isotope labels require distinct trajectory type IDs.

Particle currents always use number weights, even when the density spectrum is
neutron weighted:

.. math::

   \mathbf j(\mathbf Q,t)=\sum_j\mathbf v_j(t)e^{i\mathbf Q\cdot\mathbf r_j(t)},
   \qquad j_L=\mathbf j\cdot\hat{\mathbf Q},\qquad
   \mathbf j_T=\mathbf j-j_L\hat{\mathbf Q}.

The longitudinal spectrum uses :math:`|\widetilde j_L|^2`; the transverse
spectrum sums :math:`|\widetilde j_{T,x}|^2+|\widetilde j_{T,y}|^2+
|\widetilde j_{T,z}|^2`. It is the sum of the two transverse polarizations in
three dimensions, not their average. At :math:`Q=0` these projections are
undefined and current spectra and SEMs contain NaN. The density spectrum remains
defined and has only an elastic contribution at this point for constant weights.

Coherent/incoherent and longitudinal/transverse are different decompositions.
The latter reveals collective vibration polarization. In amorphous systems,
resolved spectral maxima can define effective dispersions; broad or overdamped
features should not automatically be interpreted as propagating phonons or
converted into phonon lifetimes.

Q coordinates and sampling
--------------------------

Cell matrices contain lattice vectors as **rows**, in ångströms (Å).
Cartesian Q and distances along a reciprocal-space path use Å⁻¹. The terminal
shows this unit as ``1/Å``. Fractional path coordinates are dimensionless.
A wave advances in phase by
:math:`2\pi` radians over one wavelength, so :math:`|\mathbf Q|=2\pi/\lambda`.
With simulation cell :math:`H`, every Q point must satisfy

.. math::

   \mathbf QH^T/(2\pi)\in\mathbb Z^3.

MDtrace validates this condition with an absolute tolerance of ``1e-5`` in
reduced box coordinates. Incompatible explicit points raise an error; they are
not silently snapped. The cell is read from the trajectory. ``dsf_cell`` can
supply a missing cell, but cannot override a different cell stored in the file.
Cells may differ by at most ``1e-5`` Å between analyzed frames. No cell
rescaling, Q folding into the first Brillouin zone, or time-dependent Q transform
is performed.

``path``
   ``dsf_q_path`` contains successive endpoints in primitive reciprocal
   coordinates; ``dsf_q_path_labels`` supplies one label per endpoint. The
   existing MDtrace commensurate-path solver selects all allowed points on each
   segment, including nonorthogonal cells and non-diagonal integer supercells.
   Endpoints themselves need not be allowed points. Empty segments raise an
   error. Plot distances use full Cartesian distances, including extended zones.

``spherical``
   Used for isotropic amorphous samples. Enumerate reciprocal-box vectors
   :math:`2\pi\mathbf nH^{-T}` within ``dsf_q_min`` and ``dsf_q_max``, excluding
   Q=0. No primitive cell is needed. Bin by magnitude in intervals of
   ``dsf_q_bin_width`` anchored at zero. First compute each vector's spectrum,
   then average spectra within a shell. Empty bins are not interpolated.

   ``dsf_max_points`` sets the total budget; ``0`` keeps all. Each allowed +/-
   pair receives a reproducible uniform random priority multiplied by its
   squared Q magnitude. Selecting the smallest priorities retains all vectors
   below the resulting cutoff radius and thins higher-Q regions. The budget
   is enforced while scanning chunks, without storing the full reciprocal
   grid. Selection is independent of the subsequent shell binning.
   The output reports both sampled and available
   vector counts, sampling fraction, shell bounds, and the sampled mean magnitude.
   The console lists each sampled shell's Q interval and used/allowed counts.
   For example, a Q range of 0 to 0.5 with a shell width of 0.05 gives
   10 radial intervals. The first covers magnitudes below 0.05, the second
   covers 0.05 up to 0.10, and the last includes the upper limit of 0.5.
   ``Q shell intervals`` counts all intervals in the requested range, including
   empty ones. ``Empty Q ranges`` lists the ranges with no allowed vectors.
   ``Sampled Q shells`` reports how many of those intervals contain sampled vectors.
   Each +/- pair contributes two vectors. Use larger budgets
   and narrower bins to check directional and radial averaging convergence.
   This averaging is not suitable for anisotropic glasses or films unless that
   angular average is explicitly intended.

   With ``dsf_save_q_resolved = 1`` or ``dsf_partials = 1``, ``.dsf_blocks``
   records actual Q ranges and standard deviations, unique
   unoriented directions (+/- and collinear multiples count as one), the three
   eigenvalues of :math:`\langle\hat Q\hat Q^T\rangle`, and effective weight count.
   Isotropic second moments have eigenvalues 1/3. A heuristic flags shells with
   fewer than six directions or eigenvalues outside [0.1, 0.6]. This is not a
   convergence test: even isotropic second moments can hide angular gaps. For
   example, an 80 x 20 x 20 Å box has only long-axis directions below
   0.30 Å⁻¹. Increasing the sampling cap cannot add absent directions.
   For a flagged shell that is only partially sampled, increase the cap and
   reassess coverage before drawing conclusions about the box. If every allowed
   vector is already included and coverage remains poor, increasing the cap
   cannot help: check cell shape/size and shell width. Changing shell width also
   changes radial averaging and must itself be checked for convergence.
   After rebinning, the sampling fraction refers only to the retained vectors,
   not to all allowed vectors in the simulation box.

Advanced explicit interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This interface is retained for existing inputs, tests and specified-direction
calculations. Students using the two templates above do not need it. The parser's
compatibility default remains ``explicit``; always set ``dsf_q_mode`` explicitly
in a new teaching input.

``explicit``
   ``dsf_qpoints`` contains triples. ``dsf_q_coordinates = fractional``
   converts them using :math:`2\pi\,\mathbf q A^{-T}`, with ``prim_unitcell``
   :math:`A`. ``cartesian`` takes the triples directly and needs no primitive
   cell. Points retain their order and direction, including distinct +/-Q.

``dsf_qpoints`` defaults to unset and ``dsf_q_coordinates`` defaults to
``fractional``. Individual explicit points are not connected or interpolated.


Normalization and statistics
----------------------------

The default ``dsf_normalization = density`` uses a two-sided periodogram per
ordinary frequency :math:`f` in THz. For an amplitude sequence :math:`x_n`
with :math:`T` frames, sampling interval :math:`\Delta t` in ps, and :math:`N` atoms,

.. math::

   P(f_k)=\frac{\Delta t}{NT}
           \left|\sum_n x_n e^{-2\pi i kn/T}\right|^2.

Thus :math:`\sum_kP(f_k)\Delta f` equals the mean squared
amplitude per atom. With no mean subtraction, the integrated coherent
spectrum is the corresponding static structure factor (including neutron
weights when requested). The time signals are transformed directly, without an
additional taper. ``dsf_subtract_mean = 1`` removes each block's
temporal mean before the FFT, so the integral describes fluctuations
instead of the full static structure factor.

Units are ``1/THz`` for number/self spectra, ``fm^2/THz`` for neutron spectra,
and ``(Å/ps)^2/THz`` for number-current spectra. The optional meV axis is
:math:`E=hf`, with :math:`h=4.135667696` meV ps. The saved intensities remain per
THz; divide them by :math:`h` when expressing densities per meV. Plotting displays
the nonnegative-frequency portion without doubling its intensity.

For comparisons with old MDtrace outputs, ``dsf_normalization = legacy`` retains
the original pynamic-compatible arbitrary normalization. For otherwise identical
blocks with :math:`T` samples,
:math:`P_\mathrm{density}=\Delta t\,T^2P_\mathrm{legacy}`.

``num_blocks`` partitions the trajectory into equal non-overlapping blocks;
trailing frames are cropped with a warning. Means and SEMs are calculated across
blocks, after directional averaging. Total neutron spectra are formed before
SEM calculation, preserving covariance between their components. One block has
undefined SEM, saved as NaN. The SEM does not include angular sampling error;
non-overlapping blocks need not be statistically independent. Frequency spacing is
:math:`1/(T\Delta t)` and the Nyquist limit is :math:`1/(2\Delta t)`.
Finite block duration still broadens peaks; check block-length convergence when
interpreting linewidths. Metadata records frequency spacing and the Nyquist limit. Raw spectra do not apply
quantum detailed-balance or instrumental corrections; resolution convolution
is an explicit post-processing step, recorded in the output history.

Input reference
---------------

Common trajectory, time-step, stride and block parameters retain their existing
meanings. ``device = cpu`` is the default; ``device = gpu`` selects one CUDA
GPU using float64/complex128. For density/current streaming, MDtrace limits its
CuPy memory pool to 90% of the available memory, taking any existing CuPy pool
limit into account. It estimates how many Q vectors fit alongside the frame
data and splits the selected vectors into as few batches as that estimate
allows. For example, if 140 vectors fit, 500 vectors run in four batches of
125. The ``GPU Q batch`` line reports the split. If an allocation fails, MDtrace
retries the unfinished Q vectors in smaller batches.
Density/current signals stay in GPU memory when they fit, or in host RAM with
GPU FFTs over full time blocks. Atom-wise self, neutron and partial calculations
reuse the same estimator with CuPy; their complete atom-wise block must fit GPU
memory. CuPy is optional and loaded only for GPU computation. GPU and CPU
results use the same normalization, transverse-polarization sum and text files.
Use ``action = compute`` to compare backends when saved data already exist.

For ``device = cpu``, ``max_cores`` limits persistent Q-point worker processes (default 4).
Each worker uses one numerical-library thread; actual process count is reduced
to fit CPU, Q-count and memory limits. Ordinary density/current DSF reads and shares small frame batches,
with at most one batch prefetched when ``trajectory_prefetch = 1`` (default).
It retains complete density/current time signals, so FFT length and frequency
resolution are unchanged. Internal batches are chosen automatically.
Each averaging block has its own progress bar, elapsed time, and estimated time
remaining. A second bar during CPU and GPU streaming counts completed Q vectors for
the displayed frames. Frame numbers run from one across the whole trajectory,
and the number after the slash is the total frame count in the input. The Q counter starts again for the next
frame batch and then tracks the FFTs for that block. A blank line separates
blocks. The block reaches 100% after its spectra have been averaged. Atom-wise calculations
instead report completed Q vectors, updating by finished Q batches in parallel.
Self spectra, nonzero neutron incoherent weights and partial outputs retain
the full-block path; currently saving Q-resolved data also requests self spectra.
No additional trajectory or intermediate result files are written. ``num_blocks`` remains a statistical setting, not a parallelism setting.
Use ``max_cores = 1`` for serial execution. Parallel execution preserves all
spectra, shell averages, block uncertainties and text output formats.
For Python scripts using multiprocessing, call the calculation inside an
``if __name__ == "__main__":`` guard, as required by the spawn start method.

.. list-table:: DSF-specific parameters
   :header-rows: 1
   :widths: 28 20 52

   * - Parameter
     - Default
     - Meaning
   * - ``density_weighting``
     - ``equal``
     - ``equal`` (unit atom weights) or ``neutron`` (scattering weights).
   * - ``current_weighting``
     - ``equal``
     - Unit atom weights for particle currents; currently the only supported choice.
   * - ``atom_types``
     - None
     - Element/isotope labels per sorted type ID; required for neutron weighting.
   * - ``dsf_q_mode``
     - ``explicit``
     - Choose ``path`` or ``spherical`` in teaching inputs. ``explicit`` is retained for advanced use.
   * - ``dsf_q_path`` / ``dsf_q_path_labels``
     - None
     - Successive fractional endpoints / one optional text label per endpoint.
   * - ``dsf_q_min`` / ``dsf_q_max``
     - ``0`` / None
     - Spherical magnitude interval in Å⁻¹; maximum is required.
   * - ``dsf_q_bin_width``
     - ``0.1``
     - Shell width in Å⁻¹.
   * - ``dsf_max_points``
     - ``5000``
     - Total Q-vector budget. ``0`` uses all; positive values must be at least 2. Odd values round down for +/- pairing.
   * - ``dsf_seed``
     - ``0``
     - Nonnegative random seed for shell subsampling.
   * - ``dsf_cell``
     - None
     - Nine row-vector cell entries in Å, when trajectory cell data are missing.
   * - ``dsf_currents``
     - ``0``
     - Set ``1`` to calculate longitudinal/transverse particle currents.
   * - ``dsf_output_components``
     - ``0``
     - Write additional coherent, incoherent and self spectrum text tables.
   * - ``dsf_output_sem``
     - ``0``
     - Write SEM tables alongside the selected spectra.
   * - ``dsf_save_q_resolved``
     - ``0``
     - Retain individual-Q means, SEMs and time blocks for independent post-processing.
   * - ``dsf_partials``
     - ``0``
     - Retain complex element-pair and species-self spectra; implies per-Q time-block retention.
   * - ``dsf_normalization``
     - ``density``
     - Frequency density or ``legacy`` arbitrary-unit normalization.
   * - ``dsf_subtract_mean``
     - ``0``
     - Remove each amplitude's block mean before the FFT.
   * - ``dsf_plot_density`` / ``dsf_plot_currents``
     - ``1`` / ``1``
     - The density switch writes one figure for S. The current switch writes three figures for total, longitudinal and transverse current. Set both to 0 to skip plotting.
   * - ``dsf_plot_frequency_max``
     - None
     - Optional positive-frequency display limit in THz.
   * - ``dsf_plot_energy_min`` / ``dsf_plot_energy_max``
     - None
     - Energy-axis limits in meV; a negative minimum shows both sides of the QENS line. Do not combine with the frequency limit.
   * - ``dsf_plot_scale``
     - ``log``
     - ``log`` shows the natural logarithm of the spectrum divided by its displayed unit.
       The three current figures share their log color scale. In ``linear`` mode each figure uses its
       own spectrum in the displayed unit to set its color scale. Both scales use the finite minimum and
       maximum in the displayed range, with automatic colorbar ticks. Only positive values contribute to the log scale.
       Rendering interpolates within contiguous occupied shells or along a crystal path.
   * - ``if_show_figures``
     - ``0``
     - Display figures interactively after saving them.

Trajectory requirements
-----------------------

Provide your own uniformly sampled GPUMD or LAMMPS production trajectory.
GPUMD XYZ uses Å and Å/fs, while canonical NetCDF uses its unit
attributes. XYZ ``Properties`` must declare ``pos:R:3`` and, for current spectra,
``vel:R:3``; other numeric fields can appear between them. Forces are never
treated as velocities. Atom IDs, when present in NetCDF, align positions,
velocities, and types across frames. Without IDs, ordering must remain fixed.
Old MDtrace text caches are regenerated when the original text trajectory is
selected. Direct use of an outdated cache is rejected, since older conversions
could misidentify velocity columns or lose requested float64 precision.

The complete teaching inputs are :ref:`dsf-student-path` and
:ref:`dsf-student-spherical`. ``action = plot`` reads only saved text and does
not require the original trajectory or Q parameters.

Output files
------------

The default number-density calculation writes three plain-text files:

``.dsf``
   Total spectrum, with frequency rows and Q-point/shell columns, matching
   pySED's table orientation. Two comment lines identify the spectrum, unit,
   atom weighting and row/column axes. For equal atom weights::

      # S(Q,f) [1/THz]; equal atom weights
      # Rows: frequency (.dsf_THz); columns: Q (.dsf_Qaxis)

``.dsf_THz``
   Signed ordinary frequency coordinates in THz. Plotting derives the energy
   coordinate using E = h*f; no redundant meV file is written.

``.dsf_Qaxis``
   One row per spectrum column: Q-axis coordinate, sampled-vector count and
   available-vector count. The first line names the three columns; Q data
   follows immediately. Short comments describe the shell grid or path ticks,
   the saved spectra and their weighting. A run ID lets ``action = thinking``
   match the trajectory and calculation settings. ``numpy.loadtxt`` reads the
   three numeric columns directly.

Plotting reads these text files directly. The density figure is ``.dsf.png``;
the current figures are ``.dsf_Ctotal.png``, ``.dsf_CL.png`` and ``.dsf_CT.png``. No NPZ is
written or read. The default crystal/amorphous examples do not retain per-Q
blocks. Earlier Q files with a JSON header or footer can still be plotted.

Additional outputs are requested explicitly:

* ``dsf_currents = 1`` writes ``.dsf_CL``, ``.dsf_CT`` and ``.dsf_Ctotal`` on the
  same frequency rows and Q columns. Each total is CL + CT. CT already sums
  both transverse components.
* Neutron mode also writes the components ``.dsf_coherent`` and ``.dsf_incoherent``.
  ``dsf_output_components = 1`` also writes these in number mode and writes
  ``.dsf_self``. The default is off.
* ``dsf_output_sem = 1`` writes ``_sem`` tables for the density spectra and
  the longitudinal and transverse current spectra.
  The default is off; block statistics are still calculated internally.
* ``dsf_save_q_resolved = 1`` or ``dsf_partials = 1`` writes one additional
  ``.dsf_blocks`` text file containing the individual Q vectors, their mapping
  to the table rows, and the per-vector spectra used for post-processing.
  Plotting reads the three-column Q table directly.

All spectrum tables use the same frequency-row/Q-column convention. Python
arrays returned by ``load_spectra`` retain the calculation's Q-first convention.
Old optional files from a previous run are ignored unless listed in the current
Q-file record; unrelated or historical results are not deleted.

Reusable spectra and scattering post-processing
----------------------------------------------------

``dsf_save_q_resolved = 1`` stores ``qblocks__NAME`` (block, individual Q,
frequency), ``qresolved__NAME`` and ``qresolved__NAME_sem`` in the optional ``.dsf_blocks`` text file. Each section begins with a comment
identifying its array name, data type and shape, followed by numeric rows.
The usual unprefixed arrays still represent output rows (Q points or shells).
Keeping blocks enables rebinning, convolution and reweighting **before** SEM
calculation. Averaging existing SEMs would lose the covariance between channels.
Storage can be substantial; per-Q retention is optional and defaults to off.

``dsf_partials = 1`` additionally saves ``pair_density``, ``species_self``,
and, with velocities, ``pair_longitudinal_current`` and
``pair_transverse_current``. The pair/species axis is between Q and frequency.
``metadata.species_labels`` and ``metadata.pair_indices`` define its ordering.
Equal weighting without element names uses ``type_ID`` labels; neutron reweighting
then requires explicit custom weight mappings. For known elements, supply
``atom_types`` at computation even when starting with ``density_weighting = equal``.

Raw pair spectra retain the full complex cross power
:math:`P_{ab}=A_a A_b^*` with the same total-atom and time normalization as the
total spectrum. Diagonal contributions are :math:`|b_a|^2P_{aa}`; each unordered
off-diagonal pair contributes :math:`2\operatorname{Re}(b_a b_b^*P_{ab})`.
Dropping the imaginary part before reweighting is incorrect for general complex
contrast. Real ``partial_coherent_sqw`` and ``partial_*_current`` contributions
sum to their totals, but individual cross terms can be negative. Incoherent
contributions are per species, weighted by :math:`|b_{a,inc}|^2`. Currents stay
number weighted under neutron reweighting. Raw complex SEMs describe RMS complex
deviations; retained blocks contain the information needed to propagate phase
and channel covariance into the final real spectrum.

The CLI does not need the trajectory and requires a new output prefix:

.. code-block:: bash

   python -m mdtrace.dsf.postprocess scattering_raw --output neutron_qens --density-weighting neutron --q-edges 0.1 0.5 1.0 1.5 2.0 3.0 --resolution-fwhm-mev 0.2 --per-mev
   mdtrace qens_plot.in

These shell widths and resolution are examples. Gaussian and Lorentzian
resolution kernels use FWHM in meV and discrete unit-area normalization. Linear
zero-padded convolution avoids frequency wraparound; intensity leaving the saved
range is lost, not silently renormalized. A warning is emitted if the requested
resolution is narrower than one energy bin. No quantum detailed-balance factor
or absolute experimental cross-section calibration is applied.

``--per-mev`` divides all spectral densities and their errors by
:math:`h=4.135667696` meV ps and updates units. Energy and frequency coordinates
are both available on loading (energy is derived from frequency). Energy-density
results plot against energy automatically;
``dsf_plot_energy_min = -2`` and ``dsf_plot_energy_max = 2`` show the two-sided
QENS region. For elastic/quasielastic analysis, retain the temporal mean unless
its removal is explicitly intended; a finite block also broadens peaks.
No automatic QENS or lifetime fit is performed.

The Python API also accepts positive per-Q averaging weights and custom complex
neutron lengths in fm:

.. code-block:: python

   from mdtrace.dsf.postprocess import (
       load_spectra, average_shells, reweight_scattering,
       apply_resolution, to_energy_density,
   )
   raw = load_spectra("scattering_raw")
   neutron = reweight_scattering(raw)  # unchanged MDtrace Sears/NIST table
   shells = average_shells(neutron, [0.1, 0.5, 1.0, 2.0, 3.0])
   measured = to_energy_density(apply_resolution(shells, 0.2))

Changing scattering weights does not change isotope masses or dynamics in the
original trajectory. Rebinning uses stored points only: it cannot undo the
original subsampling, interpolate missing directions, or prove isotropy. Empty
shells remain absent. Edges are left-closed/right-open except the last edge,
which is included; Q=0 is excluded from spherical averaging.

Validation and sources
----------------------

Tests cover complex multi-element weights against explicit time/atom-pair sums;
Parseval integrals for density, self and both current spectra; a traveling
particle satisfying :math:`\omega^2S=Q^2C_L`; a transverse oscillator; periodic
image invariance; nonorthogonal crystal paths; complete and subsampled reciprocal
shells; atom reordering; velocity units; and GPUMD conversion through plotting.
These are synthetic numerical and workflow checks, not an experimental
validation or a benchmark of an equilibrated amorphous-Si potential.

``tests/test_dsf_dynasor.py`` is an optional executable comparison with dynasor
2.5. It checks complete reciprocal spheres for orthogonal and skew cells, then
runs dynasor on two deterministic crystal-like/disordered mixed-species
trajectories. Density, self, element-pair and current spectra agree after
aligning finite-record estimators and units; zero-lag correlations independently
check the integrated strengths. To run it in an environment containing dynasor:

.. code-block:: bash

   python -m pytest tests/test_dsf_dynasor.py -q

The comparison explicitly restores the :math:`(T-\ell)/T` lag weights and folds
linear correlations into circular ones before transforming to the periodogram
grid. It does not equate raw peaks from different finite-time estimators.
The dynasor 2.5 implementation uses the **average** of two transverse
polarizations, whereas MDtrace uses their sum; its currents also use Å/fs
instead of Å/ps. Tests therefore convert longitudinal correlations by
:math:`10^6` and transverse correlations by :math:`2\times10^6`. General complex
directional cross terms are tested internally; the dynasor neutron comparison
uses both +/-Q and +/-frequency symmetrization to match its real pair convention.

The initial density FFT estimator and optional legacy convention follow
`pynamic-structure-factor <https://github.com/tyst3273/pynamic-structure-factor>`_.
Density and current correlations are established statistical-mechanical methods;
see `Fransson et al. (2021) <https://doi.org/10.1002/adts.202000240>`_ and
`Horbach et al. (2001) <https://doi.org/10.1007/s100510170299>`_. The retained
neutron data cite `Sears (1992) <https://doi.org/10.1080/10448639208218770>`_ and
the `NIST transcription <https://ncnr.nist.gov/resources/n-lengths/list.html>`_.

Calculation cost
----------------

The standard density/current workflow skips atom-wise FFTs when no self,
nonzero incoherent or species-partial contribution is needed. Requesting
``dsf_output_components``, ``dsf_save_q_resolved`` or ``dsf_partials`` retains
self spectra. Nonzero neutron incoherent weights always retain their required
atom-wise FFTs. The low-level ``block_spectra`` API still returns self spectra
by default; the workflow selects only the needed calculation. Already stacked
time-block arrays are reused for mean/SEM evaluation.
