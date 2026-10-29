SED theory and implementation
=============================

MDtrace implements the eigenvector-free spectral energy density (SED)
expression used by Thomas *et al.* [Thomas2010]_ and the PYSED article
[Liang2025]_.

Eigenvector-free SED
--------------------

For basis atom :math:`b`, repeated unit cell :math:`l`, Cartesian direction
:math:`\alpha`, and equilibrium unit-cell reference position
:math:`\mathbf r_0(l)`,

.. math::

   \Phi'(\mathbf q,\omega)
   =
   \frac{1}{4\pi\tau_0N_T}
   \sum_{\alpha=1}^{3}\sum_{b=1}^{n}m_b
   \left|
   \int_0^{\tau_0}
   \sum_{l=1}^{N_T}
   \dot u_\alpha(l,b,t)
   \exp\left[
   i\mathbf q\cdot\mathbf r_0(l)-i\omega t
   \right]dt
   \right|^2.

:math:`n` is the number of basis atoms, :math:`N_T` is the number of
repeated unit cells, :math:`m_b` is the basis-atom mass, and :math:`\tau_0`
is the duration of one trajectory block.

Discrete Fourier transform and time step
----------------------------------------

NumPy and CuPy FFTs evaluate a discrete sum. The continuous integral is
approximated by

.. math::

   \int_0^{\tau_0}v(t)e^{-i\omega t}dt
   \approx
   \Delta t
   \sum_{j=0}^{N-1}v_j e^{-i\omega t_j}.

Because SED contains the squared magnitude of this integral, the numerical
normalization contains :math:`\Delta t^2`:

.. math::

   \Phi'_\mathrm{discrete}
   =
   \frac{\Delta t^2}{4\pi\tau_0N_T}
   \sum_{\alpha,b}m_b
   \left|
   \operatorname{FFT}\left[
   \sum_l
   \dot u_\alpha(l,b,t)
   e^{i\mathbf q\cdot\mathbf r_0(l)}
   \right]\right|^2.

The PYSED article writes the continuous integral explicitly. The dynasor
implementation [Dynasor2025]_ likewise multiplies the squared FFT by
``delta_t**2``.

Internal and output units
-------------------------

MDtrace converts velocity to :math:`\mathrm{m\,s^{-1}}`, time to seconds,
and mass to kilograms during the calculation. Therefore,

.. math::

   \Delta t\sum v
   :
   \quad
   \mathrm{s}\,
   \frac{\mathrm m}{\mathrm s}
   =
   \mathrm m,

and

.. math::

   \frac{
   m_b\left|\Delta t\sum v\right|^2
   }{\tau_0}
   =
   \frac{\mathrm{kg\,m^2}}{\mathrm s}
   =
   \mathrm{J\,s}.

The symbol :math:`m_b` denotes atomic mass. The :math:`\mathrm{m^2}` in the
unit is metre squared and comes from the squared time-integrated velocity.

This internal quantity is a density per angular frequency,
:math:`\Phi_\omega`, in :math:`\mathrm{J\,s}`. MDtrace converts the completed
one-sided spectrum to a density per ordinary frequency in
:math:`\mathrm{eV/THz}`:

.. math::

   \Phi_f[\mathrm{eV/THz}]
   =
   \Phi_\omega[\mathrm{J\,s}]
   \frac{2\pi\,10^{12}}{1.602176634\times10^{-19}}.

The numerical conversion factor is approximately
:math:`3.921769\times10^{31}`. The factor :math:`2\pi` converts angular
frequency to ordinary frequency, :math:`10^{12}` converts THz to Hz, and the
denominator is the exact joule-per-electronvolt conversion.

The ``.SED`` files, partial SED files, q-point slice plots, and fitted
Lorentz/DHO peak amplitudes all use :math:`\mathrm{eV/THz}`. The dispersion heatmap
displays the dimensionless natural-log ratio
:math:`\ln\left(\Phi_f/(1\,\mathrm{eV\,THz^{-1}})\right)`. The complete
public unit and plotting conventions are collected in :doc:`sed_units`.
Consequently, integrating a written spectrum over its ``.THz`` axis directly
gives an energy in eV:

Figures retain the conventional spectral notation
:math:`\Phi(\mathbf q,\omega)`, while the numerical frequency axis is written
as ordinary frequency in THz.

.. math::

   E[\mathrm{eV}]
   =
   \int \Phi_f(f)[\mathrm{eV/THz}]\,df[\mathrm{THz}].

Energy-preserving one-sided spectrum
------------------------------------

MDtrace folds positive and negative frequencies:

.. math::

   \Phi_+(\omega_k)
   =
   \begin{cases}
   \Phi(0), & k=0,\\
   \Phi(+\omega_k)+\Phi(-\omega_k), & k>0.
   \end{cases}

This preserves two-sided spectral power while writing only non-negative
frequencies. For a classical equilibrium crystal and a commensurate Q point,

.. math::

   \int_0^\infty
   \Phi_+(\mathbf q,\omega)d\omega
   \approx
   \frac{1}{2}N_\mathrm{bands}k_BT,
   \qquad
   N_\mathrm{bands}=3n.

Because the output is in :math:`\mathrm{eV/THz}`, an effective SED
temperature can be checked directly from the area under the written spectrum:

.. code-block:: python

   import numpy as np

   k_B = 1.380649e-23
   eV_to_J = 1.602176634e-19
   prefix = "CNT"
   q_index = 1
   n_basis = 2

   frequency_thz = np.loadtxt(prefix + ".THz")
   sed = np.loadtxt(prefix + ".SED")[:, q_index]
   energy_eV = np.trapz(sed, frequency_thz)
   energy_joule = energy_eV * eV_to_J
   temperature_sed = energy_joule / (
       0.5 * (3 * n_basis) * k_B
   )
   print(f"SED temperature: {temperature_sed:.2f} K")

Finite sampling, constraints, removed center-of-mass motion, and Γ-point
translational modes cause deviations. At Γ, the three translational modes may
be absent, giving approximately :math:`(3n-3)k_BT/2`.

.. _commensurate-q-points:

Commensurate Q points
---------------------

Periodicity and coordinate conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the primitive-cell matrix as :math:`\mathbf C` and the supercell
matrix as :math:`\mathbf S`, with lattice vectors as **rows**. Their
relationship is

.. math::

   \mathbf S=\mathbf P\mathbf C,
   \qquad \mathbf P\in\mathbb Z^{3\times3},
   \qquad \det\mathbf P\ne0.

The primitive reciprocal basis includes :math:`2\pi`:

.. math::

   \mathbf B=2\pi\mathbf C^{-\mathsf T},
   \qquad \mathbf q_\mathrm{cart}=\mathbf q_\mathrm{red}\mathbf B.

Reduced coordinates are relative to this **primitive reciprocal basis**,
not to the supercell reciprocal basis. Compatibility with periodic boundary
conditions requires an integer number of phase cycles across every supercell
vector:

.. math::

   \frac{\mathbf q_\mathrm{cart}\mathbf S^{\mathsf T}}{2\pi}
   =\mathbf q_\mathrm{red}\mathbf P^{\mathsf T}
   =\mathbf n\in\mathbb Z^3.

Thus :math:`\exp(i\mathbf q_\mathrm{cart}\cdot\mathbf S_j)=1` for each
supercell vector :math:`\mathbf S_j`. The vector :math:`\mathbf n` contains
integer reciprocal-supercell indices; it is not a multiple of the atom count
or supercell size. No orthogonality assumption is used. The complete matrix
:math:`\mathbf P`, including its off-diagonal entries, is retained.

Endpoint normalization
~~~~~~~~~~~~~~~~~~~~~~

The :doc:`q_path <input_parameters/q_path>` input specifies path endpoints.
MDtrace applies one normalization policy to both the solver and its output:

* For each decimal coordinate :math:`x`, find the closest rational value
  :math:`r` whose denominator is at most 12. Use :math:`r` when
  :math:`|r-x|\le10^{-5}`; otherwise retain the decimal value as a fraction.
  For example, ``0.33333`` becomes :math:`1/3`, whereas ``0.3333`` retains
  its decimal value. This is an interpretation of the input, not an increase
  in numerical resolution.
* Exact ``fractions.Fraction`` values and fraction strings passed directly to
  ``BZPathHelper`` are preserved. The input-file parser converts coordinates
  to floating point first, so input-file fractions subsequently follow the
  decimal normalization policy above. Small-denominator fractions such as
  ``1/3`` and ``1/2`` are recommended for conventional high-symmetry points.
* The normalized endpoints are used throughout the exact solve and the
  construction of each output point. Conversion to floating point occurs
  after interpolation, when forming Cartesian q-points and output parameters.

Greatest common divisor and Bézout's identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The path solver uses the **greatest common divisor (GCD)** and
**Bézout's identity**. The **extended Euclidean algorithm** computes the
integer coefficients in that identity.

Let :math:`\mathbf s` and :math:`\mathbf e` denote the normalized endpoints.
The closed path is

.. math::

   \mathbf q_\mathrm{red}(f)=\mathbf s+f(\mathbf e-\mathbf s),
   \qquad 0\leq f\leq1,

and its representation in supercell reciprocal coordinates is

.. math::

   \mathbf a=\mathbf s\mathbf P^{\mathsf T},
   \qquad \mathbf b=(\mathbf e-\mathbf s)\mathbf P^{\mathsf T},
   \qquad \mathbf a+f\mathbf b\in\mathbb Z^3.

The solver obtains all solutions jointly, without separately enumerating
each coordinate's solutions and intersecting the resulting sets.

**1. Clear the denominators.** Let :math:`D>0` be the least common multiple
of the denominators of all components of :math:`\mathbf a` and
:math:`\mathbf b`. Then

.. math::

   \mathbf A=D\mathbf a\in\mathbb Z^3,
   \qquad \mathbf H=D\mathbf b\in\mathbb Z^3.

**2. Reduce the integer direction.** For :math:`\mathbf H\ne\mathbf0`, set

.. math::

   g=\gcd(|H_1|,|H_2|,|H_3|)>0,
   \qquad \mathbf v=\mathbf H/g.

The components of :math:`\mathbf v` have greatest common divisor one.
Bézout's identity guarantees an integer vector :math:`\mathbf w` satisfying
the relations below. Repeated application of the extended Euclidean algorithm
computes its coefficients:

.. math::

   \mathbf w\cdot\mathbf H=g,
   \qquad \mathbf w\cdot\mathbf v=1.

**3. Test whether the line intersects the integer lattice.** Define
:math:`m=\mathbf w\cdot\mathbf A`. Any integer solution :math:`\mathbf n`
must have :math:`k=\mathbf w\cdot\mathbf n\in\mathbb Z`, so

.. math::

   Dk=m+fg,
   \qquad f=\frac{Dk-m}{g},
   \qquad
   \mathbf n=\frac{\mathbf A-m\mathbf v}{D}+k\mathbf v.

Therefore the infinite line intersects the integer lattice if and only if

.. math::

   A_i-mv_i\equiv0\pmod D
   \quad\text{for every }i=1,2,3.

If this divisibility condition fails, the path has no commensurate points.
If it holds, every integer :math:`k` in the following bounds gives a solution,
and there are no other solutions.

**4. Restrict the solutions to the segment.** The condition
:math:`0\le f\le1` gives

.. math::

   k_\mathrm{min}=\left\lceil\frac{m}{D}\right\rceil,
   \qquad
   k_\mathrm{max}=\left\lfloor\frac{m+g}{D}\right\rfloor,
   \qquad f_k=\frac{Dk-m}{g}.

When :math:`k_\mathrm{min}>k_\mathrm{max}`, the line has lattice
intersections but none lie on this segment. Otherwise, the number and
spacing of the solutions are

.. math::

   N_q=k_\mathrm{max}-k_\mathrm{min}+1,
   \qquad \Delta f=\frac{D}{g},
   \qquad
   \mathbf q_{k,\mathrm{cart}}
   =[\mathbf s+f_k(\mathbf e-\mathbf s)]\mathbf B.

The divisibility checks and bounds use Python integers, and the parameters
are retained as exact fractions until output. Negative coordinates, reversed
paths, and noncommensurate endpoints are supported. A path endpoint is
included only when it satisfies the integer condition.

For example, with :math:`\mathbf P=\operatorname{diag}(12,8,4)` and
:math:`\mathbf s=(0,0,0)`, :math:`\mathbf e=(1/2,1/2,0)`, one has
:math:`D=1`, :math:`\mathbf A=\mathbf0`, :math:`\mathbf H=(6,4,0)`,
:math:`g=2`, and :math:`m=0`. The three solutions are

.. math::

   f=0,\tfrac12,1,
   \qquad
   \mathbf q_\mathrm{red}=(0,0,0),\ (1/4,1/4,0),\ (1/2,1/2,0).

Boundary cases and numerical checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :math:`\mathbf H=\mathbf0`, the normalized path has zero length. It
returns its single point at :math:`f=0` if all components of
:math:`\mathbf A` are divisible by :math:`D`; otherwise it returns no points.
Endpoint equality is tested exactly after normalization, rather than by a
floating-point ``allclose`` comparison that could discard a short segment.

``build_commensurate_path`` retains its two-array interface: Cartesian
q-points of shape ``(N, 3)`` and path parameters of shape ``(N,)``, ordered
from the supplied start toward the end. Empty results have shapes ``(0, 3)``
and ``(0,)``. Points are not folded into another Brillouin zone, and this
per-segment helper does not deduplicate endpoints shared by separate segments.

Cell matrices must be finite, nonsingular, and of shape ``(3, 3)``.
MDtrace computes :math:`\mathbf S\mathbf C^{-1}`, rounds it to an integer
matrix :math:`\mathbf P`, and requires an absolute error at most
:math:`10^{-4}` in each entry, with zero relative tolerance. A singular
rounded matrix is rejected. After conversion to Cartesian coordinates, the
returned points are also checked against
:math:`\mathbf q_\mathrm{cart}\mathbf S^{\mathsf T}/(2\pi)`: each component
must be within :math:`10^{-4}` of an integer. This final check catches
inconsistent cell precision or numerical conversion errors. The integer
solve is exact for the inferred :math:`\mathbf P` and normalized endpoints;
the supplied cell matrices and returned Cartesian arrays use floating point.

Consequences for SED sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The solver preserves the supercell-periodic spatial Fourier basis used by
SED. It supports both orthogonal and non-orthogonal cells; the reference
crystal and the chosen periodic supercell must still describe the trajectory
being analyzed. Passing a periodicity check does not make a nonperiodic
cropped region into a periodic crystal.

For a fixed supercell and path, a complete exact solver must return the same
set of commensurate points. The method based on the greatest common divisor
and Bézout's identity reduces intermediate
enumeration; it does not create extra allowed q-points or improve the
physical resolution of an existing trajectory. Denser sampling requires a
suitable larger supercell or a supercell matrix chosen for the target path.
Longer trajectories improve frequency resolution, not q-point spacing.
Arbitrary incommensurate points mix the finite cell's discrete wave vectors
and should not be treated as additional independent modes when fitting
quantitative SED linewidths.

The periodicity condition is also described in the
`Phonopy formulation <https://phonopy.github.io/phonopy/formulation.html#commensurate-points>`_.
The one-dimensional construction using the greatest common divisor and
Bézout's identity is distinct from Phonopy's
Smith-normal-form enumeration of a complete commensurate grid. For direct
input examples, see :doc:`input_parameters/q_path`.

Calculation logic
-----------------

``Phonon.py`` follows this sequence:

1. Read the basis mapping and prepare basis IDs, masses, type mappings, and Q
   points once.
2. Select serial NumPy, persistent multiprocessing, or single-GPU CuPy.
3. Read one requested trajectory block through the common text/NetCDF source.
   With ``trajectory_prefetch = 1``, prepare the next block in one background
   thread while the current block is computed.
4. Average the reference-atom positions to obtain one equilibrium unit-cell
   reference vector for each repeated cell in the block.
5. Construct :math:`\exp(i\mathbf q\cdot\mathbf R_l)`.
6. Use ``tensordot`` to contract the unit-cell axis while keeping time, Q
   point, and Cartesian direction.
7. Apply the FFT along time and multiply :math:`|\mathrm{FFT}|^2` by mass.
8. Retain type/x/y/z components or sum the total SED.
9. Apply :math:`\Delta t^2/(4\pi\tau_0N_T)` and accumulate the block online.
10. Average blocks and fold positive and negative frequencies.
11. Convert once from internal :math:`\mathrm{J\,s}` to
    :math:`\mathrm{eV/THz}`, then write the positive-frequency output.

The CPU path creates one process pool for the complete SED calculation. Each
trajectory block is copied once into shared memory, and workers receive only
small Q-point ranges. The CuPy path uploads one block, checks estimated device
memory before the first block, and frees its memory pools after completion.

For the CuPy timing report, ``Sum of the six rows above`` has a literal
meaning: it adds the six block-wait and GPU-operation rows. ``Other SED work``
accounts for the remaining setup, allocation, accumulation, averaging, and
cleanup inside the SED calculation. Their sum is printed as
``Total SED calculation time``. The final SED-step report separately shows
initial trajectory reading/setup, the SED calculation, and
output/finalization. These three stages sum to the reported
``SED compute done`` time.

Partial SED
-----------

With :math:`\mathcal B_s` denoting basis atoms of species :math:`s`,

.. math::

   \Phi'_{s,\alpha}(\mathbf q,\omega)
   =
   \frac{1}{4\pi\tau_0N_T}
   \sum_{b\in\mathcal B_s}m_b
   \left|
   \int_0^{\tau_0}
   \sum_l
   \dot u_\alpha(l,b,t)
   e^{i\mathbf q\cdot\mathbf r_0(l)-i\omega t}dt
   \right|^2.

The total is recovered by summing over species and Cartesian directions.
For example, ``SrTiO3.SED_O_y`` contains the oxygen contribution polarized
along :math:`y`. Partial files are stored in
``SrTiO3_partial_SED/`` and use the element symbol instead of an internal type
index.

SED lifetime convention
-----------------------

For an isolated peak, the fitting workflow uses one zero-background
Lorentzian or velocity-spectrum DHO over a local valley-bounded range. The
range is the complete local basin
between the lowest points of the detection-smoothed log spectrum separating
neighboring detected peaks. This prevents a small raw-data fluctuation from
truncating a broad peak to only a few fitting samples. For the first or last
detected peak, the outer boundary is the lowest smoothed
point between that peak and the selected frequency-range edge. The Lorentz
model is

.. math::

   L(f)
   =
   \frac{I}{
   1+\left[(f-f_c)/h\right]^2
   },

where :math:`h` is HWHM in THz and FWHM is :math:`2h`. The current output
reports the linewidth-derived SED lifetime convention

.. math::

   \tau_\mathrm{SED}[\mathrm{ps}]
   =
   \frac{1}{2\pi h[\mathrm{THz}]}.

For a weakly damped DHO, ``h`` has the same FWHM convention. This number is a
spectral/coherence-time convention, not automatically the energy-relaxation
time used in phonon transport. Pure dephasing, unresolved overlap, and strong
anharmonicity can broaden a line without defining one quasiparticle lifetime.

For critical or overdamped DHO fits, MDtrace writes the linewidth-derived
``tau_SED`` for completeness, but it should be treated only as a qualitative
timescale rather than a strict phonon lifetime.
MDtrace also checks whether both fitted half-maximum points lie inside the
actual local fitting range. If either side is missing, the terminal output
flags the peak as incomplete. The reported HWHM still
describes the complete analytic Lorentz or DHO line shape: the nonlinear
optimizer estimates it from the peak height, slope, and curvature available
inside the fitting range. It is therefore a model-extrapolated width rather
than a FWHM directly bracketed by measured samples on both sides. Such a width
is more sensitive to overlap, background, and line-shape assumptions, so its
reported linewidth-derived lifetime should be used qualitatively only. The fit
figure deliberately draws the model only over the data range used in the
optimization and does not display the extrapolated part.
For comparison across strongly anharmonic systems, hold the trajectory length,
frequency resolution, Q grid, detector settings, and fitting window rule fixed
and interpret linewidth/time trends qualitatively.

References
----------

.. [Liang2025] T. Liang, W. Jiang, K. Xu, H. Bu, Z. Fan, W. Ouyang, and
   J. Xu, "PYSED: A tool for extracting kinetic-energy-weighted phonon
   dispersion and lifetime from molecular dynamics simulations," *Journal of
   Applied Physics* **138**, 075101 (2025).
   https://doi.org/10.1063/5.0278798

.. [Thomas2010] J. A. Thomas, J. E. Turney, R. M. Iutzi, C. H. Amon, and
   A. J. H. McGaughey, "Predicting phonon dispersion relations and lifetimes
   from the spectral energy density," *Physical Review B* **81**, 081411
   (2010). https://doi.org/10.1103/PhysRevB.81.081411

.. [Dynasor2025] E. Berger *et al.*, "Dynasor 2: From simulation to
   experiment through correlation functions," *Computer Physics
   Communications* **316**, 109759 (2025).
   https://doi.org/10.1016/j.cpc.2025.109759
