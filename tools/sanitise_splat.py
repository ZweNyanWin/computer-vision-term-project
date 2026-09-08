"""Write a splat with the non-finite Gaussians removed, next to the original.

MLX3D's compactor selects on opacity and importance and has no finite-value
filter, so a run that produced a degenerate Gaussian exports it. Two of this
project's three runs did: 3 rows in `fast`, 146 in MCMC, carrying NaN in position,
scale and rotation. The reported `balanced` artifact has none.

Those rows never affected a score - the evaluator drops them, and re-running every
evaluation with that gate reproduced all three metrics CSVs byte-for-byte, because
a NaN position makes every comparison in the tile binner false and the Gaussian is
never binned. But an exported checkpoint is also handed to viewers, converters and
other people's tools, and "it happens to be ignored by this rasteriser" is not a
property to rely on.

So this writes `splat_clean.ply` beside the original rather than overwriting it.
The reported checkpoints keep their provenance, and the count removed is printed
and recorded so the two files can never be silently confused.

    .venv-mlx3d/bin/python tools/sanitise_splat.py model3d/gaussian/*/splat.ply
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    from mlx3d.splatting import GaussianModel

    for p in paths:
        if not p.exists():
            print(f"  skip {p} (missing)")
            continue
        model = GaussianModel.load_ply(str(p))
        n = model.num_gaussians
        finite = np.ones(n, dtype=bool)
        for value in model.params.values():
            finite &= np.isfinite(np.asarray(value).reshape(n, -1)).all(axis=1)
        dropped = int((~finite).sum())
        if dropped == 0:
            print(f"  {p}: {n:,} Gaussians, all finite - no clean copy needed")
            continue
        model.select(np.where(finite)[0])
        out = p.with_name("splat_clean.ply")
        model.save_ply(str(out))
        record = {
            "source": p.name,
            "gaussians_before": n,
            "gaussians_after": int(finite.sum()),
            "removed_non_finite": dropped,
        }
        (p.with_name("splat_clean.json")).write_text(json.dumps(record, indent=2) + "\n")
        print(f"  {p}: removed {dropped} of {n:,} -> {out.name} "
              f"({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
