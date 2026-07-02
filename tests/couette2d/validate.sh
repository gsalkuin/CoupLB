#!/usr/bin/env bash
set -euo pipefail

# Moving-wall (type 2) validation: 2D Couette flow vs analytic linear
# profile. Reference result: machine precision (max|err| ~ 1e-18).
#
# Run:
#   bash validate.sh
#   NPROCS=2 LAMMPS_BIN=/path/to/lmp bash validate.sh

LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
LMP="mpirun -n ${NPROCS} ${LAMMPS_BIN}"

rm -f couplb_couette.dat
${LMP} -in in.couette2d -log log.couette2d.lammps
python3 ../poiseuille2d/validate.py couette couplb_couette.dat 0.02
