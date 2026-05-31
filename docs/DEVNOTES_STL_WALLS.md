# CoupLB Developer Notes: STL Solid Walls

This branch adds static no-slip walls from triangulated STL geometry. The goal
is CAD-defined microfluidic channels and obstacles without changing the
manuscript-clean flat-wall path.

## Design

- `solid_stl file no-slip` reads ASCII or binary STL on each rank.
- `solid_scale` and `solid_translate` place STL coordinates in LAMMPS units.
- `solid_side inside|outside` selects whether the inside or outside of the
  closed STL surface is solid.
- Grid nodes are classified by ray parity against the STL surface.
- For each fluid node and lattice direction whose neighbor is solid, CoupLB
  stores a link-wise wall intersection fraction and wall velocity.
- Streaming checks this link table before the legacy `type[]` wall handling and
  applies interpolated no-slip bounce-back.

The current implementation intentionally loads all triangles on all ranks. That
is acceptable for validation and modest STL files; large production CAD should
add BVH/spatial-bin culling.

## Validation

`tests/stl_channel/channel_walls.stl` is a closed box that encloses the fluid
volume and uses `solid_side outside`. Its y-faces lie at `y=-0.5` and `y=31.5`,
matching the half-way bounce-back location of the legacy `wall_y 1 1` channel.

Fresh validation against `/home/exouser/coupmpm-dev/lammps/build/lmp`:

- `tests/poiseuille3d/in.poiseuille3d`: PASS, max relative error 0.0899%.
- `tests/stl_channel/in.stl_channel`: PASS, max relative error 0.0899%.
- The generated profile files are byte-identical for the two runs.

Run the STL validation with:

```bash
cd tests/stl_channel
LAMMPS_BIN=/home/exouser/coupmpm-dev/lammps/build/lmp bash validate.sh
```

## Limitations

- STL walls are 3D-only.
- Only static no-slip is supported for STL walls in this branch.
- Meshes should be closed, watertight, and non-self-intersecting.
- Curved free-slip, pressure inlet/outlet CAD boundaries, moving rigid CAD
  walls, and local triangle acceleration are follow-up work.
