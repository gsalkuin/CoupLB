#!/usr/bin/env bash
set -uo pipefail

# Run every automated CoupLB validation test and print a summary.
# Tests without a validate.sh (drag_forces_*) are qualitative and skipped;
# see tests/README.md for their manual pass criteria.
#
# Usage:
#   LAMMPS_BIN=/path/to/lmp bash run_all.sh
#   NPROCS=4 LAMMPS_BIN=/path/to/lmp bash run_all.sh
#
# Note: tests/subcycling/validate.sh is long (3 tracked runs + spinup);
# set SKIP_SLOW=1 to leave it out.

cd "$(dirname "$0")"
export LAMMPS_BIN="${LAMMPS_BIN:-lmp}"
export NPROCS="${NPROCS:-1}"
SKIP_SLOW="${SKIP_SLOW:-0}"

declare -a results=()
fail=0

for d in */; do
  d="${d%/}"
  [ -f "$d/validate.sh" ] || continue
  if [ "$SKIP_SLOW" = "1" ] && [ "$d" = "subcycling" ]; then
    results+=("SKIP  $d (SKIP_SLOW=1)")
    continue
  fi
  echo "=== $d ==="
  if (cd "$d" && bash validate.sh); then
    results+=("PASS  $d")
  else
    results+=("FAIL  $d")
    fail=1
  fi
  echo
done

echo "============================"
echo " CoupLB test suite summary"
echo "============================"
printf '%s\n' "${results[@]}"
exit ${fail}
