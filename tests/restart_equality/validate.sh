#!/usr/bin/env bash
set -uo pipefail

# Restart differential invariant:
#   checkpoint at 5000 + restart to 10000  ==  uninterrupted run 10000
# (bit-identical step-10000 profile). Checkpoints are decomposition-tied,
# so part 1 and part 2 must use the same NPROCS.
#
# Run:
#   bash validate.sh
#   NPROCS=2 LAMMPS_BIN=/path/to/lmp bash validate.sh

LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
NPROCS="${NPROCS:-1}"
LMP="mpirun -n ${NPROCS} ${LAMMPS_BIN}"

rm -f prof_ref.dat prof_part1.dat prof_restart.dat ck_restart.*.clbk

${LMP} -in in.reference -log log.reference > /dev/null 2>&1 \
  || { echo "*** FAIL: reference run failed"; exit 1; }
${LMP} -in in.part1 -log log.part1 > /dev/null 2>&1 \
  || { echo "*** FAIL: part 1 (checkpoint) run failed"; exit 1; }
${LMP} -in in.part2 -log log.part2 > /dev/null 2>&1 \
  || { echo "*** FAIL: part 2 (restart) run failed"; exit 1; }

if diff <(awk '$1==10000' prof_ref.dat) <(awk '$1==10000' prof_restart.dat) > /dev/null; then
  echo "PASS: restart continuation bit-identical to uninterrupted run at step 10000"
  exit 0
else
  echo "*** FAIL: restarted run diverges from uninterrupted run at step 10000:"
  diff <(awk '$1==10000' prof_ref.dat) <(awk '$1==10000' prof_restart.dat) | head -6
  exit 1
fi
