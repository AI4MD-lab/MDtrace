DSF for crystals and amorphous systems
======================================

The development checkout supports classical density dynamic structure factors
and longitudinal/transverse particle-current spectra. The same coordinate-based
FFT kernel applies to crystals and disordered systems; no equilibrium-site map,
``basis.in``, or phonon eigenvectors are used. This addition is separate from the
published 1.1.0 SED-only release. DSF is now the default method; SED analysis
is provided separately by pySED.

Scope and observables
---------------------

Use a fixed, fully periodic simulation cell and uniformly sampled coordinates.
Particle currents additionally require velocities from the same frames and atom
ordering. The first implementation uses NumPy and NetCDF input; text input is
converted through ``trajectory_read_mode = cache``. Variable-cell trajectories,
X-ray form-factor databases, GPU DSF, and automatic peak/lifetime fitting are not
included. The independent post-processing module supports neutron reweighting,
shell averaging, Gaussian/Lorentzian energy-resolution convolution and per-meV
spectral densities for classical neutron/QENS analysis.

``experiment = number`` computes the number-density spectrum with unit weights.
Its ``.dsf`` equals its coherent collective spectrum; the neutron-incoherent
output is zero. The separate ``.dsf_self`` is the unweighted single-particle
spectrum and is already part of the collective spectrum, so it must not be added
to it again. ``experiment = neutron`` uses the existing Sears/NIST bound neutron
table without changing its entries. Missing scattering lengths raise an error.

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

Cell matrices contain lattice vectors as **rows**, in Angstrom. Cartesian Q is
in rad/Angstrom and includes :math:`2\pi`. With simulation cell :math:`H`, every
Q point must satisfy

.. math::

   \mathbf QH^T/(2\pi)\in\mathbb Z^3.

MDtrace validates this condition with an absolute tolerance of ``1e-5`` in
reduced box coordinates. Incompatible explicit points raise an error; they are
not silently snapped. The cell is read from the trajectory. ``dsf_cell`` can
supply a missing cell, but cannot override a different cell stored in the file.
Cells may differ by at most ``1e-5`` Angstrom between analyzed frames. No cell
rescaling, Q folding into the first Brillouin zone, or time-dependent Q transform
is performed.

``explicit``
   ``dsf_qpoints`` contains triples. ``dsf_q_coordinates = fractional``
   converts them using :math:`2\pi\,\mathbf q A^{-T}`, with ``prim_unitcell``
   :math:`A`. ``cartesian`` takes the triples directly and needs no primitive
   cell. Points retain their order and direction, including distinct +/-Q.

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

   ``dsf_max_points_per_shell`` caps the sampled vectors in each shell; ``0``
   keeps all. Sampling is uniform without replacement over +/- pairs and
   reproducible with ``dsf_seed``. The output reports both sampled and available
   vector counts, shell bounds, and the sampled mean magnitude. Use larger caps
   and narrower bins to check directional and radial averaging convergence.
   This averaging is not suitable for anisotropic glasses or films unless that
   angular average is explicitly intended.

   ``.dsf_Qdiagnostics`` reports actual Q ranges and standard deviations, unique
   unoriented directions (+/- and collinear multiples count as one), the three
   eigenvalues of :math:`\langle\hat Q\hat Q^T\rangle`, and effective weight count.
   Isotropic second moments have eigenvalues 1/3. A heuristic flags shells with
   fewer than six directions or eigenvalues outside [0.1, 0.6]. This is not a
   convergence test: even isotropic second moments can hide angular gaps. For
   example, an 80 x 20 x 20 Angstrom box has only long-axis directions below
   0.30 rad/Angstrom. Increasing the sampling cap cannot add absent directions.
   Check cell shape/size as well as the sampling cap and radial bin width.

Normalization and statistics
----------------------------

The default ``dsf_normalization = density`` uses a two-sided periodogram per
ordinary frequency :math:`f` in THz. For an amplitude sequence :math:`x_n`,
window :math:`w_n`, sampling interval :math:`\Delta t` in ps, and :math:`N` atoms,

.. math::

   P(f_k)=\frac{\Delta t}{N\sum_nw_n^2}
           \left|\sum_n w_nx_n e^{-2\pi i kn/T}\right|^2.

Thus :math:`\sum_kP(f_k)\Delta f` equals the window-weighted mean squared
amplitude per atom. With no window or mean subtraction, the integrated coherent
spectrum is the corresponding static structure factor (including neutron
weights when requested). ``dsf_window = hann`` uses a periodic Hann window with
window-energy correction. ``dsf_subtract_mean = 1`` removes each block's
unweighted temporal mean before windowing, so the integral describes fluctuations
instead of the full static structure factor.

Units are ``1/THz`` for number/self spectra, ``fm^2/THz`` for neutron spectra,
and ``(Angstrom/ps)^2/THz`` for number-current spectra. The optional meV axis is
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
:math:`1/(T\Delta t)` and the Nyquist limit is :math:`1/(2\Delta t)`; windowing
also broadens peaks. Metadata records both values. Raw spectra do not apply
quantum detailed-balance or instrumental corrections; resolution convolution
is an explicit post-processing step, recorded in the output history.

Input reference
---------------

Common trajectory, time-step, stride and block parameters retain their existing
meanings. DSF uses serial NumPy; ``max_cores`` does not parallelize this module.

.. list-table:: DSF-specific parameters
   :header-rows: 1
   :widths: 28 20 52

   * - Parameter
     - Default
     - Meaning
   * - ``experiment``
     - ``neutron``
     - ``number`` or ``neutron`` density weighting; currents always use number weights.
   * - ``atom_types``
     - None
     - Element/isotope labels per sorted type ID; required for neutron weighting.
   * - ``dsf_q_mode``
     - ``explicit``
     - ``explicit``, ``path``, or ``spherical``.
   * - ``dsf_q_coordinates``
     - ``fractional``
     - ``fractional`` or ``cartesian`` for explicit input; paths use fractional coordinates.
   * - ``dsf_qpoints``
     - None
     - Explicit triples; decimal and fraction inputs are supported.
   * - ``dsf_q_path`` / ``dsf_q_path_labels``
     - None
     - Successive fractional endpoints / one optional text label per endpoint.
   * - ``dsf_q_min`` / ``dsf_q_max``
     - ``0`` / None
     - Spherical magnitude interval in rad/Angstrom; maximum is required.
   * - ``dsf_q_bin_width``
     - ``0.1``
     - Shell width in rad/Angstrom.
   * - ``dsf_max_points_per_shell``
     - ``100``
     - Positive even cap, or ``0`` for all reciprocal vectors.
   * - ``dsf_seed``
     - ``0``
     - Nonnegative random seed for shell subsampling.
   * - ``dsf_cell``
     - None
     - Nine row-vector cell entries in Angstrom, when trajectory cell data are missing.
   * - ``dsf_currents``
     - ``0``
     - Set ``1`` to calculate longitudinal/transverse particle currents.
   * - ``dsf_save_q_resolved``
     - ``0``
     - Retain individual-Q means, SEMs and time blocks for independent post-processing.
   * - ``dsf_partials``
     - ``0``
     - Retain complex element-pair and species-self spectra; implies per-Q time-block retention.
   * - ``dsf_normalization``
     - ``density``
     - Frequency density or ``legacy`` arbitrary-unit normalization.
   * - ``dsf_window``
     - ``none``
     - Rectangular (``none``) or periodic ``hann`` window.
   * - ``dsf_subtract_mean``
     - ``0``
     - Remove each amplitude's block mean before windowing.
   * - ``dsf_plot_frequency_max``
     - None
     - Optional positive-frequency display limit in THz.
   * - ``dsf_plot_energy_min`` / ``dsf_plot_energy_max``
     - None
     - Energy-axis limits in meV; a negative minimum shows both sides of the QENS line. Do not combine with the frequency limit.
   * - ``dsf_plot_scale``
     - ``log``
     - ``log`` or ``linear`` intensity color scale.
   * - ``if_show_figures``
     - ``0``
     - Display figures interactively after saving them.

Examples and execution
----------------------

Provide your own uniformly sampled GPUMD or LAMMPS production trajectory.
GPUMD XYZ uses Angstrom and Angstrom/fs, while canonical NetCDF uses its unit
attributes. XYZ ``Properties`` must declare ``pos:R:3`` and, for current spectra,
``vel:R:3``; other numeric fields can appear between them. Forces are never
treated as velocities. Atom IDs, when present in NetCDF, align positions,
velocities, and types across frames. Without IDs, ordering must remain fixed.
Old MDtrace text caches are regenerated when the original text trajectory is
selected. Direct use of an outdated cache is rejected, since older conversions
could misidentify velocity columns or lose requested float64 precision.

Crystal Si template (replace the lattice with the actual production cell's
primitive lattice in the same Cartesian orientation):

.. literalinclude:: ../../examples/dsf/crystal.in
   :language: text

Isotropic amorphous template:

.. literalinclude:: ../../examples/dsf/amorphous.in
   :language: text

Run ``mdtrace crystal.in`` or ``mdtrace amorphous.in``. ``action = thinking``
computes and plots when data are missing or numerical inputs/source metadata
change. ``action = compute`` always recomputes; ``action = plot`` reads only the
saved archive. It does not require the original trajectory or Q parameters.

Output files
------------

``.dsf``, ``.dsf_coherent``, ``.dsf_incoherent``, ``.dsf_self``
   Total, coherent, neutron-incoherent and number-self spectra. Each has an
   accompanying ``_sem`` file. Rows correspond to output Q points or shells;
   columns follow the signed ``.dsf_THz`` frequency axis.

``.dsf_CL``, ``.dsf_CT``
   Number-weighted longitudinal/transverse current spectra and ``_sem`` files.
   Written only when currents are requested.

``.dsf_Qcart``, ``.dsf_Qbox``, ``.dsf_Qfrac``
   Individual sampled Q vectors in Cartesian, reciprocal-box and primitive
   reciprocal coordinates. The fractional file is empty without a primitive
   cell. Spherical averages are not represented as fictitious individual vectors.

``.dsf_Qmap``, ``.dsf_Qaxis``
   Mapping from individual Q-vector indices to output rows, and the plot
   coordinate / sampled count / available count of each output row.

``.dsf_Qdiagnostics``
   For spherical results: Q mean, actual minimum/maximum, standard deviation,
   direction count, second-moment eigenvalues, effective weight count and
   heuristic coverage flag. After rebinning, available counts refer only to
   retained vectors, not the entire allowed reciprocal lattice.

``.dsf_metadata.json``, ``.dsf.npz``, ``.dsf.png``
   Settings, units, source signature and statistical conventions; a complete
   archive including shell bounds, path ticks and all arrays; and the plotted
   spectra. ``.dsf_meV`` supplies the energy axis, not an intensity conversion.

Reusable spectra and scattering post-processing
----------------------------------------------------

``dsf_save_q_resolved = 1`` stores ``qblocks__NAME`` (block, individual Q,
frequency), ``qresolved__NAME`` and ``qresolved__NAME_sem`` in the same archive.
The usual unprefixed arrays still represent output rows (Q points or shells).
Keeping blocks enables rebinning, convolution and reweighting **before** SEM
calculation. Averaging existing SEMs would lose the covariance between channels.
Storage can be substantial; per-Q retention is optional and defaults to off.

``dsf_partials = 1`` additionally saves ``pair_density``, ``species_self``,
and, with velocities, ``pair_longitudinal_current`` and
``pair_transverse_current``. The pair/species axis is between Q and frequency.
``metadata.species_labels`` and ``metadata.pair_indices`` define its ordering.
Number mode without element names uses ``type_ID`` labels; neutron reweighting
then requires explicit custom weight mappings. For known elements, supply
``atom_types`` at computation even when starting with ``experiment = number``.

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

   python -m mdtrace.dsf.postprocess scattering_raw.dsf.npz --output neutron_qens --experiment neutron --q-edges 0.1 0.5 1.0 1.5 2.0 3.0 --resolution-fwhm-mev 0.2 --per-mev
   mdtrace qens_plot.in

These shell widths and resolution are examples. Gaussian and Lorentzian
resolution kernels use FWHM in meV and discrete unit-area normalization. Linear
zero-padded convolution avoids frequency wraparound; intensity leaving the saved
range is lost, not silently renormalized. A warning is emitted if the requested
resolution is narrower than one energy bin. No quantum detailed-balance factor
or absolute experimental cross-section calibration is applied.

``--per-mev`` divides all spectral densities and their errors by
:math:`h=4.135667696` meV ps and updates units. Energy and frequency coordinates
are both retained. Energy-density archives plot against energy automatically;
``dsf_plot_energy_min = -2`` and ``dsf_plot_energy_max = 2`` show the two-sided
QENS region. For elastic/quasielastic analysis, retain the temporal mean unless
its removal is explicitly intended; a finite block/window also broadens peaks.
No automatic QENS or lifetime fit is performed.

The Python API also accepts positive per-Q averaging weights and custom complex
neutron lengths in fm:

.. code-block:: python

   from mdtrace.dsf.postprocess import (
       load_spectra, average_shells, reweight_scattering,
       apply_resolution, to_energy_density,
   )
   raw = load_spectra("scattering_raw.dsf.npz")
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
polarizations, whereas MDtrace uses their sum; its currents also use Angstrom/fs
instead of Angstrom/ps. Tests therefore convert longitudinal correlations by
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
