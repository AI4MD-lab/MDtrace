# MDtrace TODO

## Spherical DSF sampling

- [ ] Reuse the atomic sums and complex FFT for each sampled pair of opposite Q vectors
  when computing equal-weight density and particle-current spectra. For 1,000 paired
  vectors, compute 500 representatives and recover each partner's power from
  \(P(-\mathbf Q,f)=P(\mathbf Q,-f)\). Count both vectors in their original shell.
  Compare the density, longitudinal, transverse and total current spectra and the
  runtime against the full calculation on the same trajectory block.

## Phonon lifetime convention

- [ ] Review and finalize the lifetime definition derived from the fitted
  Lorentzian linewidth.
  - The fitted parameter is the ordinary-frequency HWHM \(h\), with
    \(\mathrm{FWHM}=2h\).
  - The current output uses \(1/(2\pi h)\), matching dynasor's
    amplitude/correlation lifetime convention.
  - The energy-relaxation lifetime commonly used in phonon transport is
    \(1/(4\pi h)\) when pure dephasing is negligible.
  - Before changing the output, decide whether to report the linewidth and
    explicitly labelled \(T_1\) and \(T_2\) values.

## Optional mode-projected SED (planned; not on the current mainline)

- [ ] Add an eigenvector-projected SED path while retaining the existing
  eigenvector-free total SED as the default:

  ```text
  sed_projection = total | mode
  eigenvector_file = band.yaml
  ```

  - Read phonon eigenvectors, q-points, harmonic frequencies, and optional
    group velocities from a phonopy-compatible ``band.yaml`` file.
  - Validate primitive/supercell atom mapping, species ordering, q-point
    matching, eigenvector normalization, and phase conventions before the
    projection is evaluated.
  - Compute the mass-weighted normal-mode velocity
    \(\dot Q_{\mathbf q\nu}(t)\) and its SED without applying artificial
    broadening, tail clipping, or normalization that changes the linewidth.
  - Treat zero-padding only as frequency-grid interpolation, not as improved
    physical frequency resolution.
  - Preserve separate output for every \((\mathbf q,\nu)\) mode and flag
    projected spectra that remain multi-peaked or strongly asymmetric because
    of finite-temperature mode mixing.
  - Add synthetic-mode and sum-rule tests, plus CPU/CuPy parity tests, before
    connecting mode-projected spectra to the lifetime-fitting workflow.
