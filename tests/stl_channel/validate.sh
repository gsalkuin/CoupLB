#!/usr/bin/env bash
set -euo pipefail

# STL channel validation, two parts:
#  1. Grid-aligned STL channel vs analytic Poiseuille (validate.py).
#  2. STL translated +0.3 dx in y, so wall links carry sub-cell distances
#     (lambda = 0.2 lower wall, 0.8 upper wall). The flow must match the
#     parabola with walls at the SHIFTED positions — this is the test that
#     the Bouzidi interpolation places the wall at the true STL surface
#     rather than at grid planes (reference result: 0.17% max error).
#
# Run:
#   bash validate.sh
#   NPROCS=4 LAMMPS_BIN=/path/to/lmp bash validate.sh

LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
LMP="mpirun -n ${NPROCS} ${LAMMPS_BIN}"

rm -f couplb_stl_channel.dat couplb_stl_shifted.dat

${LMP} -in in.stl_channel -log log.stl_channel.lammps
python3 ../poiseuille2d/validate.py poiseuille couplb_stl_channel.dat 5e-6

${LMP} -in in.stl_channel_shifted -log log.stl_shifted.lammps
python3 - <<'EOF'
import sys
import numpy as np

rows = [l.split() for l in open('couplb_stl_shifted.dat')
        if not l.startswith('#') and l.strip()]
last = max(int(r[0]) for r in rows)
d = np.array([[float(x) for x in r] for r in rows if int(r[0]) == last])
y, u = d[:, 2], d[:, 4]

g, nu, H = 5e-6, 0.1, 32.0
umax = g * H * H / (8 * nu)
ylo = -0.5 + 0.3                     # lower wall plane after solid_translate
yhat = (y - ylo) / H
uex = 4 * umax * yhat * (1 - yhat)
m = (yhat > 0) & (yhat < 1)
err = np.max(np.abs(u[m] - uex[m])) / umax

print(f"  shifted STL (step {last}): u_max={u.max():.6e} exact={umax:.6e} "
      f"max rel err={100*err:.3f}%")
if err < 0.02:
    print("  -> PASS (walls resolved at sub-cell STL positions)")
else:
    print("  -> FAIL: error exceeds 2% — Bouzidi lambda interpolation broken?")
    sys.exit(1)
EOF
