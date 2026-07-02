# CoupLB Test Suite

Run everything automated:

```bash
LAMMPS_BIN=/path/to/lmp bash run_all.sh          # serial
NPROCS=4 LAMMPS_BIN=/path/to/lmp bash run_all.sh # 4 MPI ranks
SKIP_SLOW=1 ... bash run_all.sh                  # skip the long subcycling test
```

Each test directory is self-contained; run one with
`cd <dir> && LAMMPS_BIN=... bash validate.sh`. Only the COUPLB package is
required (examples/ need more — see the top-level README).

## Tests

There are two kinds: **physics validations** compare against analytic
solutions with a tolerance; **differential invariants** assert exact
equalities the implementation must satisfy bit-for-bit — these are the
tests that catch state-handoff and parallelization bugs that outcome
tests self-heal past (see `docs/DEVNOTES_2026-07-02.md`).

| Test | Kind | Validates | Reference result |
|:-----|:-----|:----------|:-----------------|
| `poiseuille2d` | physics | D2Q9 + no-slip walls + gravity vs analytic parabola | 0.85% max err |
| `poiseuille3d` | physics | D3Q19 channel vs analytic parabola | <1% max err |
| `couette2d` | physics | moving wall (type 2) vs linear profile | machine precision |
| `stl_channel` | physics | STL no-slip walls, grid-aligned AND shifted +0.3 dx (sub-cell Bouzidi lambda) | 0.09% / 0.17% |
| `subcycling` | physics | `md_per_lb` 4 and 10 track the md_per_lb=1 baseline (slow: multi-phase) | tolerance per script |
| `run_continuation` | invariant | `run 5000; run 5000` == `run 10000` | bit-identical |
| `restart_equality` | invariant | checkpoint at 5000 + restart == uninterrupted run | bit-identical |
| `drag_forces_point` | manual | IBM point drag: terminal velocity, no transverse drift | criteria printed by deck |
| `drag_forces_sphere` | manual | IBM marker-sphere drag | criteria printed by deck |
| `drag_forces_sphere_vtk` | manual | as above, with VTK output enabled | inspect output |

## Manual pass criteria (drag tests)

The drag decks print their checks at the end of the run:
v_x positive and steady over the last ~2000 steps; v_y, v_z at machine
zero; constant mass and Ma < 0.1 in the CoupLB diagnostics; no density
clamp warnings.

## Conventions

- `LAMMPS_BIN` — path to a LAMMPS binary built with the COUPLB package
  (default `lmp`).
- `NPROCS` — MPI rank count (default 1). All automated tests pass at 1
  and 4 ranks; `restart_equality` requires the same rank count for both
  phases (checkpoints are decomposition-tied).
- A validator exits nonzero on failure, so `run_all.sh` is CI-ready.
