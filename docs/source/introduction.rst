Introduction
============

MDtrace analyzes density fluctuations and particle currents in molecular-dynamics
trajectories. The same DSF calculation supports crystal Q paths and spherical
averages for isotropic amorphous systems.

The current implementation provides coherent, incoherent and self spectra,
longitudinal/transverse number-current spectra, element-pair decomposition,
neutron reweighting, and independent spectral post-processing. See
:doc:`dsf_workflow` for definitions, units, limitations and validation.

Use a fixed, fully periodic cell and uniformly sampled frames. Velocities are
needed only for currents. DSF uses NumPy with optional Q-point multiprocessing and reads native NetCDF or
text trajectories converted to a NetCDF cache. EELS remains planned.

SED computation, peak fitting and phonon-lifetime output are provided separately
by `pySED <https://pysed.readthedocs.io/>`_. This development change does not
modify existing trajectories or previously calculated SED results.
