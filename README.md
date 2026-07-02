# CoupLB: Coupled Lattice Boltzmann Package for LAMMPS

Lattice Boltzmann Method (LBM) coupled with Molecular Dynamics via the
Immersed Boundary Method (IBM). Runs as a standard LAMMPS `fix` with full
MPI parallelism and optional OpenMP threading.

## Quick Start

```lammps
units           lj
dimension       2
boundary        p f p
atom_style      atomic

region          box block 0 8 0 32 -0.5 0.5
create_box      1 box
create_atoms    1 single 4 0.5 0

group           noatoms empty
mass            1 1.0
pair_style      none
timestep        1.0

fix flow noatoms couplb 8 32 1 0.1 1.0 &
    wall_y 1 1 &
    gravity 5e-6 0.0 0.0 &
    check_every 2000

run 50000
```

## Features

- **Lattices:** D2Q9 (2D) and D3Q19 (3D)
- **Collision:** BGK with Guo forcing scheme, OpenMP-parallel
- **IBM coupling:** Roma (3-point) or Peskin4 (4-point) delta kernels,
  sub-stepping with momentum conservation
- **Boundaries:** No-slip, moving wall, free-slip, open, periodic
- **CAD walls:** Static no-slip STL walls via link-wise bounce-back
- **External forcing:** Constant or LAMMPS variable-style body forces
- **I/O:** VTK (.vti + .pvd), solid VTK (.vtp), binary checkpoint/restart,
  ASCII profiles
- **Parallelism:** MPI domain decomposition with three ghost-exchange types

## Build

C++14, MPI. No external dependencies.

CoupLB is an external package, so LAMMPS must be told it exists before
`yes-couplb` / `-DPKG_COUPLB` will work:

1. Copy the `CoupLB/` source files into `lammps/src/COUPLB/`.
2. Register the package name:
   - **CMake:** add `COUPLB` to the `set(STANDARD_PACKAGES ...)` list in
     `lammps/cmake/CMakeLists.txt` (alphabetical, near `COLLOID`).
     Without this, `-DPKG_COUPLB=yes` is **silently ignored**.
   - **Traditional make:** add `couplb` to the `PACKAGE` list in
     `lammps/src/Makefile`.

Then:

```bash
cd lammps/src
make yes-couplb
make -j8 mpi
```

Or CMake:
```bash
mkdir build && cd build
cmake ../cmake -DPKG_COUPLB=yes -DBUILD_MPI=yes
make -j8
```

The validation tests under `tests/` need only the COUPLB package. The
`examples/` decks additionally require the MOLECULE, BPM, and DIPOLE
packages (`-DPKG_MOLECULE=yes -DPKG_BPM=yes -DPKG_DIPOLE=yes`).

## Syntax

```
fix ID group couplb Nx Ny Nz nu rho0 [keywords...]
```

| Argument | Description |
|:---------|:------------|
| `Nx Ny Nz` | LBM grid dimensions |
| `nu` | Kinematic viscosity (LAMMPS units) |
| `rho0` | Reference fluid density (LAMMPS units) |

Common keywords: `md_per_lb`, `xi_ibm`, `gravity`, `wall_x/y/z`, `wall_vel`,
`solid_stl`, `solid_scale`, `solid_translate`, `solid_side`, `kernel`, `vtk`,
`vtk_solid`, `vtk_region`, `checkpoint`, `restart`, `check_every`.

Full syntax, keyword table, and extended examples: **[docs/keywords.md](docs/keywords.md)**

## Documentation

| Document | Contents |
|:---------|:---------|
| [docs/keywords.md](docs/keywords.md) | Full syntax reference, keyword table, examples |
| [docs/architecture.md](docs/architecture.md) | Source layout, grid design, time integration, MPI |
| [docs/theory.md](docs/theory.md) | Unit conversion, IBM coupling, stability, accuracy |
| [docs/parallelism.md](docs/parallelism.md) | MPI decomposition, ghost exchange, OpenMP |
| [docs/io.md](docs/io.md) | VTK, solid output, profiles, checkpoint/restart |
| [docs/DEVNOTES_2026-05-30.md](docs/DEVNOTES_2026-05-30.md) | Notes on setup, periodic IBM, PVD time, and profile fixes |
| [docs/DEVNOTES_2026-07-02.md](docs/DEVNOTES_2026-07-02.md) | Pre-release review: run-continuation bug root cause, lessons learned |

## IBM Coupling (overview)

```
F_particle = ξ × m × (u_fluid − v_particle) / dt
F_grid     = −F_particle / force_scale
```

All atoms in the fix group are coupled. Sub-stepping (`md_per_lb > 1`) scales
grid force by 1/N for exact momentum conservation over one LBM interval.

See **[docs/theory.md](docs/theory.md)** for full details.

## Numerical Guidelines

- τ must be > 0.5 (best in [0.6, 1.5])
- Ma = u_max / cs < 0.3 (warned), < 0.5 (error)
- IBM bodies: diameter/dx ≥ 10 recommended
- VTK output: keep frequency low for grids > 128³ (gathers to rank 0)

## Known Limitations

- BGK collision only (no MRT/entropic stabilization)
- Single-pass IBM (no iterative no-slip correction)
- Checkpoint tied to processor count and decomposition
- VTK gathers to rank 0 (scalability limit for very large grids)
- Fixed Lagrangian marker volume
- Single ghost layer: IBM delta stencils are clipped (and renormalized,
  momentum-conserving) for particles within dx/2 below a subdomain's upper
  face or periodic wrap; measured ~0.06% relative force error in smooth flow
- STL Bouzidi bounce-back is not exactly mass-conserving for interior
  obstacles with impinging flow (planar channels conserve exactly);
  monitor with `check_every` and resolve obstacles with >= 10 cells
- Fluid state is preserved across `run` commands only if the domain
  decomposition and timestep are unchanged (otherwise reset, with warning)

## Examples

```
examples/
├── oscillating-flow/   — oscillating body force with MD particles
├── swimming/           — IBM-coupled swimmer with custom lam read
tests/
├── run_all.sh          — run the whole automated suite (see tests/README.md)
├── poiseuille2d/       — Poiseuille flow validation
├── poiseuille3d/       — 3D channel validation
├── couette2d/          — moving-wall Couette validation (machine precision)
├── stl_channel/        — STL no-slip channel, grid-aligned + sub-cell shifted
├── drag_forces_point/  — IBM drag on point particle
├── drag_forces_sphere/ — IBM drag on sphere
├── drag_forces_sphere_vtk/ — with VTK output
├── subcycling/         — sub-stepping validation
├── run_continuation/   — `run N; run M` == `run N+M` bit-exact (differential invariant)
└── restart_equality/   — checkpoint+restart == uninterrupted run bit-exact
```

Run the automated suite with `bash tests/run_all.sh` (details and per-test
reference results: [tests/README.md](tests/README.md)).

## TODO

- [ ] MRT / entropic collision for high-Re flows
- [ ] Iterative IBM correction for no-slip enforcement
- [ ] Handle edge/corner nodes for mixed-wall configurations
- [ ] Early exit on VTK file open failure
- [ ] Re-add `nsub` (multiple LBM per MD) if applications require it

## References

Alkuino, Gabriel, Joel T. Clemmer, Christian D. Santangelo, and Teng Zhang.
"Bonded-particle model for magneto-elastic rods." arXiv preprint
arXiv:2603.27279 (2026).
