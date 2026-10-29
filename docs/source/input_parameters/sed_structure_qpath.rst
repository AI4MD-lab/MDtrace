Structure and Q path
====================

These parameters connect trajectory atoms to a primitive-cell basis and build
the exact Q points compatible with the finite MD supercell.

.. list-table::
   :class: parameter-index
   :header-rows: 1
   :widths: 29 17 54

   * - Parameter
     - Default
     - Purpose
   * - :doc:`basis_lattice_file <basis_lattice_file>`
     - ``basis.in``
     - Map each atom to a repeated cell, basis atom, and mass.
   * - :doc:`prim_unitcell <prim_unitcell>`
     - ``None``
     - Set three primitive lattice vectors in Angstrom.
   * - :doc:`prim_axis <prim_axis>`
     - ``None``
     - Transform a simulation-cell basis to the desired primitive basis.
   * - :doc:`supercell_dim <supercell_dim>`
     - ``1 1 1``
     - Set primitive-cell repetitions and Q-point resolution.
   * - :doc:`rescale_prim <rescale_prim>`
     - ``1``
     - Reconstruct the primitive cell from a relaxed trajectory cell.
   * - :doc:`num_qpaths <num_qpaths>`
     - ``1``
     - Set the number of connected path segments.
   * - :doc:`q_path_name <q_path_name>`
     - ``GA``
     - Provide one label for each path vertex.
   * - :doc:`q_path <q_path>`
     - ``None``
     - Provide ``num_qpaths + 1`` reduced-coordinate triples.

Example cubic path:

.. code-block:: text

   basis_lattice_file = ../structure/basis.in
   supercell_dim       = 20 20 20
   prim_unitcell       = 3.89598 0 0  0 3.89598 0  0 0 3.89598
   rescale_prim        = 1

   num_qpaths  = 3
   q_path_name = GXMG
   q_path      = 0 0 0  0 1/2 0  1/2 1/2 0  0 0 0

See :doc:`../sed_workflow/preparing_trajectory_basis` for basis-file
preparation and :doc:`../theory` for commensurate-Q construction.

:doc:`Back to SED parameters <sed>`

.. toctree::
   :hidden:

   basis_lattice_file
   prim_unitcell
   prim_axis
   supercell_dim
   rescale_prim
   num_qpaths
   q_path_name
   q_path
