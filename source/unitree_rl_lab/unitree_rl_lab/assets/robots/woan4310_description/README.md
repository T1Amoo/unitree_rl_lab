# WOAN4310 robot description provenance

This directory contains the minimum robot-description asset set needed by the
`Unitree-WOAN4310-Velocity` Isaac Lab task:

- one URDF (`urdf/dog_V2.urdf`);
- 21 STL visual meshes referenced by that URDF.

Source repository: <https://github.com/XUEHAIXU/WOAN_GYM>

Source commit: `e80f82df1ac22384a194810102aaf661f7fa3461`

Retrieved: 2026-08-24

The files are copied byte-for-byte from
`resources/robots/dog_V2/{urdf/dog_V2.urdf,meshes/*.STL}`. See `SHA256SUMS`
for the integrity record. ROS launch files, build files, logs, CSV files, and
training code from the source repository are intentionally not included.

## License status

The legal rights holder confirmed on 2026-08-24 that the one URDF and 21 STL
files listed in `SHA256SUMS` are released under the Apache License, Version 2.0
(`Apache-2.0`). See [`ASSET_LICENSE.md`](ASSET_LICENSE.md) for the exact scope
and the repository-root [`LICENCE`](../../../../../../LICENCE) for the full
license terms.

The source repository's `setup.py` declares `BSD-3-Clause` for its Python
package, but source commit `e80f82d` does not contain a root or asset-specific
license file. That historical metadata is retained here as provenance; the
asset-specific Apache-2.0 grant above records the rights holder's explicit
authorization for this vendored asset set.

Keep this provenance note, `ASSET_LICENSE.md`, and `SHA256SUMS` with the assets
when redistributing them.
