#!/usr/bin/env bash
set -euo pipefail

LAMMPS_BIN="${LAMMPS_BIN:-lmp}"

"${LAMMPS_BIN}" -in in.stl_channel -log log.stl_channel.lammps
python3 ../poiseuille2d/validate.py poiseuille couplb_stl_channel.dat 5e-6
