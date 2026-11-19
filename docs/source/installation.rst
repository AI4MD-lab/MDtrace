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

The required NumPy, SciPy, netCDF4, Matplotlib, and Seaborn dependencies are
installed automatically.

DSF execution
-------------

The current DSF calculation uses serial NumPy. CUDA and CuPy are not required.
Installing CuPy does not enable a DSF GPU backend. This development checkout
retains its own trajectory readers and does not require pySED to be installed.
