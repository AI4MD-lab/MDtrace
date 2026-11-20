<p align="center">
  <img src="docs/source/_static/logo.svg" alt="MDtrace logo" width="420">
</p>

<h1 align="center">MDtrace</h1>

<p align="center">
  Molecular Dynamic Trajectory Trace — tracing phonon and vibrational physics in MD trajectories.
</p>

<p align="center">
  <a href="https://mdtrace.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/mdtrace/badge/?version=latest" alt="Documentation status">
  </a>
  <img src="https://img.shields.io/badge/version-1.1.0-3f7f6f" alt="MDtrace version 1.1.0">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.10-3776ab" alt="Python 3.10 or newer">
</p>

### Why the name?

```
 Molecular        Dynamic        Trajectory    Trace
    │                │               │            │
    ▼                ▼               ▼            ▼
   [MD]        +   [MD]        +  [tra]   +   [ce]
    │                │               │            │
    └────────────────┘               └────────────┘
         MD                               trace

                    ──▶  MDtrace  ◀──
```

The name is self-contained in the phrase: **M**olecular **D**ynamic **Tra**jectory **Trace**. The `MD` captures the simulation method, and `trace` is the action — tracing phonons, tracing spectra, tracing the physics hidden in every atom's path.

This branch computes the density dynamic structure factor and longitudinal and
transverse current spectra for crystals and amorphous systems with a fixed
periodic cell. Density spectra require atomic positions. Current spectra also
require velocities saved in the same frames.

## Installation

Use Python 3.10 or newer. To install the DSF version in a fresh Conda environment:

```bash
conda create -n mdtrace python=3.12 -y
conda activate mdtrace
git clone --branch codex/remove-sed https://github.com/AI4MD-lab/MDtrace.git
cd MDtrace
python -m pip install .
mdtrace -h
```

If you already have a Python environment, start at `git clone`. Pip installs
the required packages, including `pysed-phonon` for trajectory I/O. The default
`device = cpu` is ready to use after installation.

**Optional NVIDIA GPU support.** With CUDA 12.x, run
`python -m pip install cupy-cuda12x`. With CUDA 13.x, use
`python -m pip install cupy-cuda13x` instead. Set `device = gpu` in your
MDtrace input file to use the GPU.

## First DSF calculation

Copy the [amorphous input](examples/dsf/amorphous.in) or the
[crystal input](examples/dsf/crystal.in) to your calculation directory as
`input.in`. Set `trajectory_file`, `time_step` (fs) and `output_data_stride`
(MD steps between saved frames) for your trajectory. For a crystal, also set
the primitive cell and reciprocal-space path. From that directory, run:

```bash
mdtrace input.in
```

The templates use `action = thinking` to compute and plot the spectra.
To redraw saved spectra after changing plot settings, use `action = plot`.
See the [DSF workflow](docs/source/dsf_workflow.rst) for input and output details.

<details>
<summary>Earlier SED workflow (release 1.1.0)</summary>

The [1.1.0 release](https://github.com/AI4MD-lab/MDtrace/tree/1.1.0)
computes and visualizes spectral energy density (SED) and fits phonon peaks.
The following commands and examples apply to that release.

## Highlights

- Builds exact wave vectors commensurate with a finite MD supercell.
- Reads GPUMD extended XYZ, one-file LAMMPS custom dumps, and compatible
  NetCDF trajectories through one block-oriented interface.
- Either streams text directly once or converts it to a reusable
  `.mdtrace.nc` cache; optional one-block prefetch overlaps I/O with SED work.
- Computes SED with serial NumPy, multiprocessing NumPy, or one optional CuPy
  GPU backend.
- Writes total and element/Cartesian-resolved SED in `eV/THz`.
- Produces phonon-dispersion maps and logarithmic single-Q spectra.
- Detects peaks with a local, dimensionless noise-significance criterion.
- Fits every peak independently with a zero-background Lorentz or
  velocity-spectrum DHO line shape; `auto` selects between them using AICc.
- Exports per-Q and combined frequency-lifetime data plus a summary figure.

## Documentation

The complete manual is available on
[Read the Docs](https://mdtrace.readthedocs.io/en/latest/).

## Command line

```bash
mdtrace                  # input.in, then the legacy input_SED.in
mdtrace input.in         # explicit input file
mdtrace -h               # command summary
```

The calculation is controlled inside the input file:

```ini
action = thinking        # run the next missing SED stage
method = sed             # supported 1.1.0 method
backend = numpy          # numpy or cupy
```

For an explicit and reproducible workflow, use the same `input.in` and change
only `action`:

1. `action = compute` — read the trajectory and overwrite the numerical SED.
2. `action = plot` — plot existing `.SED`, `.Qpts`, and `.THz` files.
3. `action = fit` — detect and fit peaks in existing SED data.

`thinking` mode is a convenience that checks existing outputs and runs the
next missing stage. MDtrace resolves relative trajectory and `basis.in` paths
relative to the input file, so the command may be launched from another
directory. Relative output prefixes and the `Fitting-Qpoint/` and `Lifetime/`
directories are written relative to the current working directory.

For spectral fitting in `thinking` mode, `lorentz_fit_all_qpoint = 1` fits all
Q points when the combined lifetime file is missing. If the parameter is
omitted or set to `0`, MDtrace fits only the zero-based Q point selected by
`qpoint_slice_index` when that Q-point lifetime file is missing. Existing
output is treated as complete; use `action = fit` to force a refit after
changing fitting parameters.

## Complete single-Q refit example

The following SrTiO3 input refits zero-based Q-point `23` after an earlier
all-Q fit. It assumes that `SrTiO3.SED`, `SrTiO3.Qpts`, `SrTiO3.THz`, and the
existing per-Q files under `Lifetime/` are already present. After replacing the
Q-point result, `re_output_total_freq_lifetime = 1` rebuilds the combined file
and the frequency-lifetime figure.

```ini
# ==================== Control ====================
action      = fit       # fit existing SED data
method      = SED
backend     = cupy      # affects compute; fitting itself uses SciPy on CPU

# ==================== MD simulation ====================
num_atoms          = 40000
total_num_steps    = 300000
time_step          = 1       # fs
output_data_stride = 15      # dump_exyz stride in gpumd_run/run.in

# ==================== Input / output ====================
trajectory_file    = ../gpumd_run/dump.xyz
basis_lattice_file = ../structure/basis.in
out_files_name     = SrTiO3
trajectory_read_mode = cache  # cache | direct for text trajectories
trajectory_prefetch  = 1      # default: prepare one block ahead
netcdf_compression_level = 1  # 0 disables compression of the text cache

# ==================== Structure ====================
supercell_dim = 20 20 20
prim_unitcell = 3.89598 0 0  0 3.89598 0  0 0 3.89598
rescale_prim  = 1             # reconstruct the NPT-rescaled primitive cell

# ==================== Q-points ====================
num_qpaths  = 5
q_path_name = GXMGRM
q_path      = 0.0 0.0 0.0  0.0 0.5 0.0  0.5 0.5 0.0  0.0 0.0 0.0  0.5 0.5 0.5  0.5 0.5 0.0

# ==================== Computation ====================
num_blocks     = 5
max_cores      = 8
output_partial = 1            # used during compute; fit does not regenerate them

# ==================== SED plot ====================
# plot_partial_SED = Sr y     # uncomment for the Sr y component
plot_cutoff_freq   = 25.0
plot_interval      = 5.0
qpoint_slice_index = 23       # zero-based index
plot_slice         = 1
if_show_figures    = 0        # save without opening an interactive window
# colorbar_min     = -20      # optional natural-log color limits
# colorbar_max     = -2

# ==================== Spectral fitting ====================
lorentz_fit_all_qpoint        = 0
lorentz_fit_freq_min          = 0
lorentz_fit_freq_max          = 25
peak_min_significance         = 5.0
fitting_function              = auto   # auto | lorentz | dho
initial_guess_hwhm            = 0.01
re_output_total_freq_lifetime = 1      # rebuild existing all-Q results
```

Run it with:

```bash
mdtrace input.in
```

For the initial all-Q fit, first use:

```ini
action                           = fit
lorentz_fit_all_qpoint           = 1
re_output_total_freq_lifetime    = 0  # all-Q fitting rebuilds the summary automatically
```

Inspect the figures under `Fitting-Qpoint/` before treating the collected
linewidths as reliable.

## Compute backends

| Configuration | SED execution |
|---|---|
| `backend = numpy`, `max_cores = 1` | Serial NumPy |
| `backend = numpy`, `max_cores > 1` | Persistent CPU workers sharing each velocity block |
| `backend = cupy` | One GPU per MDtrace process |

`num_blocks` controls spectral averaging and the amount of trajectory data
loaded per block. The requested number of saved frames must be divisible by
`num_blocks`.

## Main outputs

| Output | Description |
|---|---|
| `<name>.SED` | Total kinetic-energy-weighted SED in `eV/THz` |
| `<name>.Qpts` | Reduced Q-point coordinates |
| `<name>.THz` | Ordinary-frequency axis in THz |
| `<name>.Q_distances_and_labels` | Path distances and high-symmetry labels |
| `<name>-SED.png` | SED dispersion map |
| `SED-<q>-qpoint.png` | Optional single-Q SED plot |
| `<name>_partial_SED/` | Optional element/Cartesian SED files and figures |
| `Fitting-Qpoint/Fitting-<q>-qpoint.png` | Per-Q fitted line shapes |
| `Lifetime/Fitting-<q>-qpoint.Fre_lifetime` | Two-column fitted frequency (THz) and `tau_SED` (ps) for one Q point |
| `Lifetime/Fitting-All-Qpoints.Fre_lifetime` | The same two columns concatenated over all Q points; no Q-index column |
| `Fitting-Frequency-Lifetime.png` | All-Q frequency-lifetime summary |

For a selected partial SED component, `Fitting-Frequency-Lifetime.png` is
stored under `<name>_partial_SED/`. The numbered fitting figures and lifetime
tables still use the shared `Fitting-Qpoint/` and `Lifetime/` directories, so
save or move an existing total-SED fit before fitting a partial component if
both results must be retained.

## Interpreting fitted lifetimes

MDtrace reports the linewidth-derived convention

```text
tau_SED = 1 / (2*pi*HWHM)
```

with HWHM in THz and `tau_SED` in ps. For isolated, underdamped peaks this is
the conventional SED linewidth lifetime. Overlapping, incomplete, critically
damped, and overdamped features should be interpreted qualitatively. In
strongly anharmonic systems, always inspect the fitted line shape and compare
only calculations with consistent trajectory length, frequency resolution,
Q grid, and fitting settings.

## Roadmap

The 1.1.0 release deliberately focuses on a clean SED workflow. Planned work
includes:

- further DSF extensions: X-ray weighting and peak analysis,
- electron energy-loss spectroscopy (EELS),
- optional mode-projected SED using external eigenvectors.

Roadmap items are not part of the supported 1.1.0 command interface.

</details>

## Citation

If MDtrace is used for SED analysis, please cite:

1. T. Liang, W. Jiang, K. Xu, H. Bu, Z. Fan, W. Ouyang, and J. Xu,
   “[PYSED: A tool for extracting kinetic-energy-weighted phonon dispersion and
   lifetime from molecular dynamics simulations](https://doi.org/10.1063/5.0278798),”
   *Journal of Applied Physics* **138**, 075101 (2025).
2. J. A. Thomas, J. E. Turney, R. M. Iutzi, C. H. Amon, and
   A. J. H. McGaughey,
   “[Predicting phonon dispersion relations and lifetimes from the spectral
   energy density](https://doi.org/10.1103/PhysRevB.81.081411),”
   *Physical Review B* **81**, 081411 (2010).

Questions and feedback: `liangting.zj@gmail.com`.

## License

MDtrace is distributed under the
[GNU General Public License v3.0 or later](LICENSE).
