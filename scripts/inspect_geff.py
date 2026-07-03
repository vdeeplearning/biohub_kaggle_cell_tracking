from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Biohub .geff sparse labels.")
    parser.add_argument("geff_path", type=Path, help="Path to a .geff directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum number of centroid rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = zarr.open_group(args.geff_path, mode="r")

    node_ids = np.asarray(root["nodes/ids"])
    t = np.asarray(root["nodes/props/t/values"])
    z = np.asarray(root["nodes/props/z/values"])
    y = np.asarray(root["nodes/props/y/values"])
    x = np.asarray(root["nodes/props/x/values"])

    points = np.stack([node_ids, t, z, y, x], axis=1)
    order = np.lexsort((node_ids, z, t))
    points = points[order]

    print(f"GEFF: {args.geff_path}")
    print(f"nodes: {len(points)}")
    print(f"t range: {t.min()}..{t.max()}")
    print(f"z range: {z.min()}..{z.max()}")
    print()
    print("labels per timepoint:")
    for time_value in np.unique(t):
        count = int(np.sum(t == time_value))
        print(f"  t={int(time_value):3d}: {count}")

    print()
    print("node_id,t,z,y,x")
    for row in points[: args.limit]:
        print(",".join(str(int(v)) for v in row))

    if len(points) > args.limit:
        print(f"... {len(points) - args.limit} more")


if __name__ == "__main__":
    main()
