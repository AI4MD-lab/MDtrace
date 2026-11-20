Installation
============

MDtrace 1.1.0 requires Python 3.10 or newer.

From source
-----------

.. code-block:: bash

   git clone https://github.com/Tingliangstu/MDtrace.git mdtrace
   cd mdtrace
   python -m pip install .
   mdtrace -h

The required ``pysed-phonon>=3.1,<4``, NumPy, SciPy, netCDF4, Matplotlib and
Seaborn dependencies are installed automatically once the required pySED release
is available. For local development, install the matching pySED checkout into
the same Python environment first:

.. code-block:: powershell

   python -m pip install -e E:/Github/pySED
   python -m pip install -e E:/Github/MDtrace
   python -c "import pySED; print(pySED.__file__)"

The printed path should point to the intended local checkout. No source-path
modifications or copied reader files are needed.

DSF execution
-------------

DSF uses NumPy with optional CPU multiprocessing controlled by ``max_cores``.
CUDA and CuPy are not required for this default backend. For a CUDA GPU, install
one CuPy package matching the installed CUDA major version, then set
``device = gpu``. For example, choose either ``cupy-cuda12x`` or
``cupy-cuda13x`` for the corresponding CUDA version. CuPy is loaded only when
GPU computation is requested. MDtrace reuses pySED's CUDA availability check
and trajectory I/O; DSF estimation remains in MDtrace.
