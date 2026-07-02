#!/usr/bin/env bash
set -euo pipefail
LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
rm -f couplb_poiseuille3d.dat
mpirun -n "${NPROCS}" "${LAMMPS_BIN}" -in in.poiseuille3d -log log.lammps
python3 ../poiseuille2d/validate.py poiseuille couplb_poiseuille3d.dat 5e-6
