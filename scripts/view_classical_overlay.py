from __future__ import annotations

import argparse
import csv
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr


VOXEL_SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay classical detections/tracks on a Biohub volume.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--geff", type=Path, default=None)
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--contrast-min", type=float, default=None)
    parser.add_argument("--contrast-max", type=float, default=None)
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--show-node-ids", action="store_true")
    return parser.parse_args()


def read_nodes(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [
            {
                "node_id": int(row["node_id"]),
                "track_id": int(row["track_id"]),
                "t": int(row["t"]),
                "z": int(row["z"]),
                "y": int(row["y"]),
                "x": int(row["x"]),
                "score": float(row["score"]),
            }
            for row in csv.DictReader(f)
        ]


def read_gt_points(geff_path: Path) -> np.ndarray:
    root = zarr.open_group(geff_path, mode="r")
    t = np.asarray(root["nodes/props/t/values"])
    z = np.asarray(root["nodes/props/z/values"])
    y = np.asarray(root["nodes/props/y/values"])
    x = np.asarray(root["nodes/props/x/values"])
    return np.stack([t, z, y, x], axis=1)


def main() -> None:
    args = parse_args()
    image = da.from_zarr(zarr.open(args.zarr_path / args.array_path, mode="r"))
    nodes = read_nodes(args.result_dir / "nodes.csv")

    viewer = napari.Viewer()
    try:
        viewer.dims.axis_labels = ("t", "z", "y", "x")
    except Exception:
        pass

    image_kwargs = {
        "name": args.zarr_path.stem,
        "scale": VOXEL_SCALE_TZYX,
        "blending": "additive",
    }
    if args.contrast_min is not None or args.contrast_max is not None:
        image_kwargs["contrast_limits"] = (
            0.0 if args.contrast_min is None else args.contrast_min,
            float(image.max().compute()) if args.contrast_max is None else args.contrast_max,
        )
    viewer.add_image(image, **image_kwargs)

    pred_points = np.asarray([[n["t"], n["z"], n["y"], n["x"]] for n in nodes], dtype=float)
    track_ids = np.asarray([n["track_id"] for n in nodes], dtype=int)
    node_ids = np.asarray([n["node_id"] for n in nodes], dtype=int)

    point_kwargs = {
        "name": "classical detections",
        "scale": VOXEL_SCALE_TZYX,
        "size": args.point_size,
        "face_color": "track_id",
        "features": {"track_id": track_ids, "node_id": node_ids},
        "opacity": 0.85,
    }
    if args.show_node_ids:
        point_kwargs["text"] = {"string": "{node_id}", "size": 8, "color": "white"}
    viewer.add_points(pred_points, **point_kwargs)

    track_data = np.asarray([[n["track_id"], n["t"], n["z"], n["y"], n["x"]] for n in nodes], dtype=float)
    if len(track_data) > 0:
        viewer.add_tracks(
            track_data,
            name="classical tracks",
            scale=VOXEL_SCALE_TZYX,
            tail_width=2,
            tail_length=8,
        )

    if args.geff is not None:
        gt_points = read_gt_points(args.geff)
        viewer.add_points(
            gt_points,
            name="sparse GEFF labels",
            scale=VOXEL_SCALE_TZYX,
            size=args.point_size * 1.8,
            face_color="magenta",
            opacity=1.0,
        )

    napari.run()


if __name__ == "__main__":
    main()
