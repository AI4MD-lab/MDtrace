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

MDtrace computes the dynamic structure factor $S(q, f)$ and longitudinal and
transverse current spectra from molecular dynamics trajectories of crystals and
amorphous materials.

## Installation

```bash
conda create -n mdtrace python=3.12 -y
conda activate mdtrace
git clone https://github.com/AI4MD-lab/MDtrace.git
cd MDtrace
python -m pip install .
mdtrace -h
```

CPU is the default. For an NVIDIA GPU, install the CuPy package for your CUDA
version (e.g. `python -m pip install cupy-cuda12x` for CUDA 12.x) and set
`device = gpu` in the input file.

## Run an amorphous sample

1. Copy the [amorphous input template](examples/dsf/amorphous.in) into your calculation directory as `input.in`. Set `trajectory_file` to your fixed-cell, periodic trajectory. Density spectra need positions. For current spectra (`dsf_currents = 1`), include velocities in the same frames.
2. Set `time_step` in fs and `output_data_stride` to the number of MD steps between saved frames. For example, a 1 fs step with a frame saved every 5 steps gives 5 fs between frames.
3. Set `action = thinking`. From the calculation directory, run:

```bash
mdtrace input.in
```

**SiO₂ training files.** From the supplied `dsf/` directory, run `mdtrace input_dsf.in`. Use `action = plot` to redraw the supplied spectra, or `action = thinking` to compute them when needed.

## Read the spectra

Open the longitudinal and transverse maps side by side. At the same wavevector magnitude $|q|$, compare the frequencies of their peaks.

| Figure suffix | What it shows |
| --- | --- |
| `.dsf_CL.png` | Atomic motion parallel to the wavevector |
| `.dsf_CT.png` | Atomic motion perpendicular to the wavevector |
| `.dsf_Ctotal.png` | Total current spectrum, $C_L + C_T$ |
| `.dsf.png` | Density fluctuations, $S(q, f)$ |

The training files use the prefix `sio2`, for example `sio2.dsf_CT.png`.
For a crystal, use the [crystal input template](examples/dsf/crystal.in).
[DSF workflow and parameters](docs/source/dsf_workflow.rst)

<details>
<summary>Why the name?</summary>

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

</details>

Questions and feedback: `liangting.zj@gmail.com`.

[GNU General Public License v3.0 or later](LICENSE)
