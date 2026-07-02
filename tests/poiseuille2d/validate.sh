#!/usr/bin/env bash
set -euo pipefail
LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
rm -f couplb_poiseuille.dat
mpirun -n "${NPROCS}" "${LAMMPS_BIN}" -in in.poiseuille2d -log log.lammps
python3 validate.py poiseuille couplb_poiseuille.dat 5e-6
