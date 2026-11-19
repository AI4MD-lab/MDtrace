"""Small synthetic input shared by MDtrace integration tests."""

GPUMD_TEXT = """\
2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:vel:R:3
Sr 0 0 0 0.001 0.002 0.003
Ti 1 1 1 0.004 0.005 0.006
2
Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:vel:R:3
Sr 0.1 0 0 0.002 0.003 0.004
Ti 1.1 1 1 0.005 0.006 0.007
"""


LAMMPS_TEXT = """\
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 20
0 30
ITEM: ATOMS id type x y z vx vy vz
2 2 2 2 2 4 5 6
1 1 1 1 1 1 2 3
ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 20
0 30
ITEM: ATOMS id type x y z vx vy vz
1 1 1.1 1 1 2 3 4
2 2 2.1 2 2 5 6 7
"""
