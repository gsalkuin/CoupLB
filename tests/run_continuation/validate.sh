#!/usr/bin/env bash
set -uo pipefail

# Run-continuation differential invariant:
#   run 5000; run 5000  ==  run 10000   (bit-identical fluid state)
#
# This is an exact equality the implementation must satisfy, not a physics
# tolerance: the fluid grid must be preserved across `run` command
# boundaries. Guards against the init()-reallocation bug fixed 2026-07-02
# (see docs/DEVNOTES_2026-07-02.md). Also checks that no unexpected
# "fluid state reset" warning fires on an unchanged decomposition.
#
# Run:
#   bash validate.sh
#   NPROCS=4 LAMMPS_BIN=/path/to/lmp bash validate.sh

LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
LMP="mpirun -n ${NPROCS} ${LAMMPS_BIN}"

rm -f prof_whole.dat prof_split.dat

${LMP} -in in.whole -log log.whole > /dev/null 2>&1 \
  || { echo "*** FAIL: in.whole did not run"; exit 1; }
${LMP} -in in.split -log log.split > /dev/null 2>&1 \
  || { echo "*** FAIL: in.split did not run"; exit 1; }

fail=0

if grep -q "fluid state reset" log.split; then
  echo "*** FAIL: unexpected fluid-state reset at run boundary"
  fail=1
fi

if diff <(awk '$1==10000' prof_whole.dat) <(awk '$1==10000' prof_split.dat) > /dev/null; then
  echo "PASS: step-10000 profile bit-identical (run 5000+5000 == run 10000)"
else
  echo "*** FAIL: split run diverges from whole run at step 10000:"
  diff <(awk '$1==10000' prof_whole.dat) <(awk '$1==10000' prof_split.dat) | head -8
  fail=1
fi

exit ${fail}
