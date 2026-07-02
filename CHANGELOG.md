# Changelog

## 2026-07-02

### Fixed
- **Fluid state was silently reset to quiescent equilibrium at every `run`
  command boundary** (`init()` reallocated the grid unconditionally). The grid
  is now preserved across `run` commands when the decomposition and timestep
  are unchanged; `run N; run M` now matches `run N+M` bit-exactly for pure
  fluid runs. A reset (with a screen warning) still occurs if the
  decomposition or timestep changes between runs.
- Grid forces are cleared in `setup()` so a run continuation does not
  double-apply the previous run's final IBM force spread.
- Stability check (`check_every`) now hard-errors on non-finite fluid
  mass/momentum. Previously a NaN field reported `Ma=0.0000` because
  max-reductions ignore NaN, so corrupted runs continued silently.
- `xi_ibm` validation corrected: the stability-relevant quantity is the
  per-substep relaxation `xi_ibm/md_per_lb`, so the bound is now
  `0 < xi_ibm <= md_per_lb` (was `<= 1`, which rejected the shipped
  `examples/swimming` deck that uses `xi_ibm 5` with `md_per_lb 167`).
- Examples: create the `vtk/` output directory (`shell mkdir -p vtk`);
  previously both examples aborted at step 0 in a fresh checkout because
  git does not track the empty directory.

### Added
- `tests/run_continuation/`: differential-invariant regression test
  asserting `run 5000; run 5000` is bit-identical to `run 10000`.
  Verified to FAIL on the pre-fix code and PASS after the fix, serial
  and 4-rank.
- `tests/stl_channel/in.stl_channel_shifted`: STL channel translated
  +0.3 dx so wall links carry sub-cell distances (lambda 0.2/0.8);
  validates that Bouzidi interpolation places walls at the true STL
  surface (0.17% max error vs shifted-wall parabola). The grid-aligned
  case alone cannot see lambda errors. `validate.sh` runs both.
- Characterized STL mass conservation: planar channels (aligned and
  shifted) conserve mass to machine precision; interior obstacles with
  impinging flow leak ~0.02%/1000 steps at 2-cell resolution (known
  interpolated-bounce-back property; documented in theory.md/README).
- `tests/couette2d/`: moving-wall (type 2) validation vs analytic linear
  profile — machine precision. Wall type 2 previously had no test.
- `tests/restart_equality/`: checkpoint + restart == uninterrupted run,
  bit-identical (differential invariant).
- `tests/run_all.sh` + `tests/README.md`: one-command suite runner
  (CI-ready, nonzero exit on failure) and per-test documentation;
  `poiseuille2d`/`poiseuille3d` gained validate.sh wrappers.
- `CITATION.cff`: GitHub citation metadata pointing at the accepted paper
  (arXiv:2603.27279; update DOI when the journal version appears).

### Documentation
- `docs/DEVNOTES_2026-07-02.md`: root-cause analysis of the run-boundary
  reset (LAMMPS `init()` is per-run, not per-fix; state creation was not
  idempotent), why the test suite missed it, and lessons learned.
- `docs/architecture.md`: new "Fix Lifecycle and Run Continuation" section.
- `docs/theory.md`: xi_ibm allowed range and penalty-coupling mass-ratio
  stability guidance.
- `docs/parallelism.md`: documented upper-face IBM stencil clipping with
  the single ghost layer.
- README build section: document that `COUPLB` must be registered in
  `cmake/CMakeLists.txt` (or `src/Makefile`) before `-DPKG_COUPLB=yes` /
  `make yes-couplb` works, and that the examples need MOLECULE+BPM+DIPOLE.
- `keywords.md`: updated `xi_ibm` bounds.
- `LICENSE`: GPL-2.0 (matching LAMMPS, whose license governs code compiled
  into it) and a README license section; `CITATION.cff` license field.

### Known limitations (documented, unchanged)
- IBM delta-kernel stencils are clipped (and renormalized) for particles
  within `dx/2` below a subdomain's upper face or a periodic wrap, because
  the grid has a single ghost layer. Measured effect ~6e-4 relative force
  error in a smooth-flow test; totals remain momentum-conserving. A second
  ghost layer would remove this.

## 2026-04-29

### Fixed
- Validate `md_per_lb >= 1` and wall types in `[0,4]` at parse time (constructor).
- Remove duplicate `enforce_wall_ghost_fields` call in `do_ibm_sub_coupling`.
- Re-evaluate `ibm_has_particles` every MD substep so late-arriving particles couple immediately rather than waiting for the next LBM cycle.
- Cache delta kernel weights in `IBM::spread()` to avoid redundant evaluations.

### Documentation
- Rewrite `README.md` as a compact landing page.
- Add user-facing docs: `architecture.md`, `io.md`, `keywords.md`, `parallelism.md`, `theory.md`.
- Add `CODE_REVIEW_2026-04-29.md` and `INDEPENDENT_VERIFICATION_2026-04-29.md`.

### Notes
- `couplb_io.h`: B1 (`error->one` vs `error->all`) retained as-is pending future collective restructuring. See verification doc for rationale.
