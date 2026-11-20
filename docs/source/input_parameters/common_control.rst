Control
-------

These parameters select what MDtrace should do and which implementation it
should use.

.. list-table::
   :class: parameter-index
   :header-rows: 1
   :widths: 28 18 54

   * - Parameter
     - Default
     - Purpose
   * - :doc:`action <action>`
     - ``thinking``
     - Select automatic workflow, forced computation, or plotting.
   * - :doc:`method <method>`
     - ``dsf``
     - Select the analysis method.
   * - :doc:`device <device>`
     - ``cpu``
     - Select NumPy CPU computation or optional CuPy GPU computation.

Minimal control block:

.. code-block:: text

   action  = thinking
   method  = dsf
   device  = cpu

:doc:`Back to Common parameters <common>`

.. toctree::
   :hidden:

   action
   method
   device
