# CoupLB Architecture

## Source Files

| File | Description |
|:-----|:------------|
| `fix_couplb.cpp` / `.h` | Main fix class: time integration, keyword parsing, IBM coupling, I/O scheduling |
| `couplb_lattice.cpp` / `.h` | Grid storage (SoA layout), macroscopic field computation, lattice descriptors (D2Q9, D3Q19) |
| `couplb_collision.h` | BGK collision with Guo source term |
| `couplb_streaming.h` | Streaming step, MPI ghost exchange (distributions, velocities, forces) |
| `couplb_boundary.h` | Wall type and velocity assignment |
| `couplb_solid_stl.h` | ASCII/binary STL reader, mesh classification, link-wise wall table construction |
| `couplb_ibm.h` | IBM interpolation (fluid → particle) and force spreading (particle → fluid) |
| `couplb_io.h` | VTK writer (gathered to rank 0), PVD time series, binary checkpoint I/O |

## Template Design

Uses `template <typename Lattice>` throughout — `D2Q9` and `D3Q19` are compile-time
lattice descriptors with `D, Q, cs2, e[Q][3], w[Q], opp[Q], reflect[D][Q]`.
The compiler generates separate 2D and 3D code paths with zero runtime dispatch
overhead.

## Grid Layout

SoA (Structure of Arrays) with ghost-inclusive indexing: `f[q*ntotal + n]`.
Ghost layers are 1 cell deep. Separate arrays for `rho, ux, uy, uz, fx, fy, fz,
type, wall_dim, bc_ux/y/z`. STL walls additionally populate per-link arrays
`wall_link_bc/dist/ux/uy/uz` indexed as `q*ntotal+n`.

---

## Time Integration Sequence

### LBM Step (once per `md_per_lb` MD steps)

```
1. apply_external_force    — add gravity body force to grid
2. compute_macroscopic     — compute ρ, u from f_i (with Guo correction)
3. collide                 — BGK collision with Guo source term
4. exchange                — MPI ghost exchange of distributions
5. stream                  — pull-scheme streaming with flat or STL link-wise bounce-back
6. enforce_wall_ghost_fields — reset wall node ρ, u to prescribed values
7. clear_forces            — zero grid force arrays for next cycle
```

### IBM Coupling (every MD step)

For `md_per_lb = 1`:
```
1. enforce_wall_ghost_fields
2. compute_macroscopic (without Guo correction)
3. exchange_velocity
4. For each particle: interpolate → compute F → apply to particle → spread to grid
5. exchange_forces (reverse communication)
```

For `md_per_lb > 1`:
```
Sub-step 0 (first): LBM step, then IBM coupling with exchange_velocity
Sub-steps 1..N-1: IBM coupling only (re-interpolate with existing fluid field)
Sub-step N-1 (last): exchange_forces after spreading
```

## Fix Lifecycle and Run Continuation

LAMMPS calls `Fix::init()` at the start of **every** `run` command, not once
per fix lifetime. CoupLB therefore separates two kinds of work inside
`init()`/`setup_grid()`:

- **Re-derived every run:** unit-conversion scales (`dt_LBM`, `vel_scale`,
  `force_scale`), relaxation time τ, boundary re-marking, wall-flag
  precomputation. These depend on `timestep`, which the user may change
  between runs.
- **Created once:** the grid itself and the distribution functions f_i — the
  primary state of the fluid, which cannot be reconstructed from atom data.

`setup_grid()` preserves the existing fluid state when the local grid
dimensions, offsets, and `dt_LBM` are unchanged since the previous run, so

```
run 5000
run 5000
```

is bit-identical to `run 10000` (regression-tested in
`tests/run_continuation/`). If the domain decomposition or the timestep
changes between runs, the fluid is re-initialized to quiescent equilibrium
and a `CoupLB WARNING` is printed — there is no physically meaningful way to
map the old populations onto a new layout (use `checkpoint`/`restart` with
an unchanged layout to carry state across separate LAMMPS invocations).

`setup()` (called after `init()`, before the first timestep of each run)
clears the grid force arrays before spreading the initial IBM forces, so a
run boundary does not double-count the previous run's final force spread.

Historical note: before 2026-07-02 the grid was unconditionally reallocated
and set to equilibrium in every `init()`, silently resetting the fluid at
every `run` boundary. See `docs/DEVNOTES_2026-07-02.md` for the root-cause
analysis and the lessons drawn from it.

## MPI Communication (three types)

| Exchange | Direction | What | When |
|----------|-----------|------|------|
| Distribution | owner→ghost | f_i after streaming | Each LBM step |
| Velocity | owner→ghost | ρ, u before IBM | Each IBM sub-step |
| Reverse force | ghost→owner (accumulate) | IBM forces at boundaries | After IBM spreading |

Reverse force exchange is critical for correctness: IBM spreading can deposit
forces on ghost cells when a particle's delta stencil extends across a
subdomain boundary. Without accumulation back to the owning rank, those forces
are silently lost.
