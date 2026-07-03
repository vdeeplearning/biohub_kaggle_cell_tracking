from __future__ import annotations

import argparse
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr


VOXEL_SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a Biohub Cell Tracking .zarr volume in napari."
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to a sample .zarr directory, such as data/train/sample.zarr.",
    )
    parser.add_argument(
        "--geff",
        type=Path,
        default=None,
        help="Optional matching .geff directory. If provided, centroid labels are overlaid.",
    )
    parser.add_argument(
        "--array-path",
        default="0",
        help="Array path inside the .zarr store. Competition samples use 0.",
    )
    parser.add_argument(
        "--contrast-min",
        type=float,
        default=None,
        help="Optional lower contrast limit.",
    )
    parser.add_argument(
        "--contrast-max",
        type=float,
        default=None,
        help="Optional upper contrast limit.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Displayed size for GEFF centroid points.",
    )
    return parser.parse_args()


def open_zarr_array(zarr_path: Path, array_path: str) -> da.Array:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Could not find {zarr_path}")

    store_path = zarr_path / array_path
    if not store_path.exists():
        raise FileNotFoundError(f"Could not find array path {store_path}")

    z = zarr.open(store_path, mode="r")
    return da.from_zarr(z)


def load_geff_points(geff_path: Path) -> np.ndarray:
    if not geff_path.exists():
        raise FileNotFoundError(f"Could not find {geff_path}")

    root = zarr.open_group(geff_path, mode="r")
    t = np.asarray(root["nodes/props/t/values"])
    z = np.asarray(root["nodes/props/z/values"])
    y = np.asarray(root["nodes/props/y/values"])
    x = np.asarray(root["nodes/props/x/values"])
    return np.stack([t, z, y, x], axis=1)


def main() -> None:
    args = parse_args()
    image = open_zarr_array(args.zarr_path, args.array_path)

    print(f"Opening {args.zarr_path}")
    print(f"shape={image.shape}, dtype={image.dtype}, chunks={image.chunksize}")

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

    if args.geff is not None:
        points = load_geff_points(args.geff)
        print(f"Overlaying {len(points)} GEFF centroid points from {args.geff}")
        if len(points) > 0:
            first = points[0].astype(int)
            unique_t = np.unique(points[:, 0].astype(int))
            print(f"First centroid is at t={first[0]}, z={first[1]}, y={first[2]}, x={first[3]}")
            print(f"Centroid timepoints: {unique_t[:30].tolist()}")
        viewer.add_points(
            points,
            name=f"{args.geff.stem} centroids",
            scale=VOXEL_SCALE_TZYX,
            size=args.point_size,
            face_color="magenta",
            opacity=1.0,
        )
        if len(points) > 0:
            try:
                viewer.dims.current_step = tuple(points[0].astype(int).tolist())
            except Exception:
                pass

    napari.run()


if __name__ == "__main__":
    main()
