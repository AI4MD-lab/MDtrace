# Crystal and amorphous DSF

These are input templates for your own production trajectories, not bundled MD
simulations. Run `mdtrace crystal.in` or `mdtrace amorphous.in` from this directory
after providing the named trajectory. Relative trajectory paths are relative to
the input file; relative output prefixes are relative to the working directory.

- Save positions and velocities at the same uniform sampling interval. GPUMD
  extended XYZ must declare `pos:R:3` and `vel:R:3` in `Properties`. Native NetCDF
  with cell and velocity unit metadata is also supported.
- Use a fixed, fully periodic simulation cell. For a crystal, match
  `prim_unitcell` to its actual size and orientation in the trajectory.
- Set `dsf_currents = 0` for a positions-only density DSF calculation.
- `experiment = number` studies number-density fluctuations. To use the
  retained neutron table, set `experiment = neutron` and supply `atom_types` in
  ascending trajectory type-ID order. GPUMD XYZ caches assign those IDs in
  species' first-appearance order. For example, a file beginning with O then Si
  needs `atom_types = O Si`, not alphabetical order.
- `dsf_subtract_mean = 1` analyzes fluctuations after removing each block's
  temporal mean; use `0` to retain the static/elastic contribution.
- Spherical averaging assumes an isotropic material. Use explicit Cartesian Q
  vectors or a crystal path for directional analysis.

The default normalized spectra are two-sided densities per THz. Current spectra
are number weighted and the transverse output sums the two perpendicular
polarizations. Q=0 has no longitudinal/transverse direction and is saved as NaN
in current outputs. `thinking` computes and plots; `action = plot` can regenerate
the figure from the `.dsf.npz` archive without loading a trajectory.

See [the DSF workflow](../../docs/source/dsf_workflow.rst) for equations, all
parameters, output definitions, limitations, and validation coverage.

## Reuse the same trajectory calculation

The crystal/amorphous templates retain per-Q time blocks (`dsf_save_q_resolved = 1`).
`scattering.in` additionally retains complex element-pair spectra and species self
spectra (`dsf_partials = 1`, which implies per-Q retention). All of these arrays
are in the `.dsf.npz` archive; the table of scattering lengths is unchanged.
Storage grows with the number of blocks, Q vectors, frequency bins and species
pairs. Disable per-Q retention for compact one-off calculations.

After `mdtrace scattering.in`, for example:

```sh
python -m mdtrace.dsf.postprocess scattering_raw.dsf.npz --output neutron_qens --experiment neutron --q-edges 0.1 0.5 1.0 1.5 2.0 2.5 3.0 --resolution-fwhm-mev 0.2 --per-mev
mdtrace qens_plot.in
```

The shell edges and resolution above are examples, not recommendations for every
material. Check the actual directions and Q spread in `.dsf_Qdiagnostics`, and use
an energy resolution supported by the trajectory's block duration. Rebinning only
uses retained Q vectors; it cannot restore omitted directions or make a coarse
box isotropic. The original raw archive is preserved. The broadened result is
classical: no quantum detailed-balance correction, absolute cross-section
calibration or automatic QENS/lifetime fitting is applied.
