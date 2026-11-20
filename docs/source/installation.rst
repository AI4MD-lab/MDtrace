Installation
============

The DSF branch requires Python 3.10 or newer.

From source
-----------

.. code-block:: bash

   conda create -n mdtrace python=3.12 -y
   conda activate mdtrace
   git clone --branch codex/remove-sed https://github.com/AI4MD-lab/MDtrace.git
   cd MDtrace
   python -m pip install .
   mdtrace -h

If you already have a Python environment, start at the clone step. Pip installs
the required dependencies, including ``pysed-phonon>=3.0`` for trajectory I/O.

DSF execution
-------------

DSF uses NumPy with optional CPU multiprocessing controlled by ``max_cores``.
CUDA and CuPy are not required for this default backend. For a CUDA GPU, install
one CuPy package matching the installed CUDA major version, then set
``device = gpu``. For example, choose either ``cupy-cuda12x`` or
``cupy-cuda13x`` for the corresponding CUDA version. CuPy is loaded only when
GPU computation is requested. MDtrace reuses pySED's CUDA availability check
and trajectory I/O; DSF estimation remains in MDtrace.
