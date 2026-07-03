from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a cropped 2D movie centered on sparse GEFF labels."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to a sample .zarr directory.")
    parser.add_argument("geff_path", type=Path, help="Path to the matching .geff directory.")
    parser.add_argument(
        "--array-path",
        default="0",
        help="Array path inside the .zarr store. Competition samples use 0.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Square crop size around each labeled centroid. Use 0 for the full 2D slice.",
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
    return parser.parse_args()


def load_points(geff_path: Path) -> np.ndarray:
    root = zarr.open_group(geff_path, mode="r")
    node_ids = np.asarray(root["nodes/ids"])
    t = np.asarray(root["nodes/props/t/values"])
    z = np.asarray(root["nodes/props/z/values"])
    y = np.asarray(root["nodes/props/y/values"])
    x = np.asarray(root["nodes/props/x/values"])
    points = np.stack([node_ids, t, z, y, x], axis=1).astype(int)
    order = np.lexsort((points[:, 0], points[:, 2], points[:, 1]))
    return points[order]


def crop_2d(frame: np.ndarray, center_y: int, center_x: int, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    y0 = center_y - half
    y1 = y0 + crop_size
    x0 = center_x - half
    x1 = x0 + crop_size

    src_y0 = max(y0, 0)
    src_y1 = min(y1, frame.shape[0])
    src_x0 = max(x0, 0)
    src_x1 = min(x1, frame.shape[1])

    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    crop = np.zeros((crop_size, crop_size), dtype=frame.dtype)
    crop[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
    return crop


def main() -> None:
    args = parse_args()
    if args.crop_size < 0:
        raise ValueError("--crop-size must be >= 0")
    if args.crop_size > 0 and args.crop_size % 2 != 0:
        raise ValueError("--crop-size must be even")

    image = zarr.open(args.zarr_path / args.array_path, mode="r")
    points = load_points(args.geff_path)
    if len(points) == 0:
        raise ValueError(f"No GEFF points found in {args.geff_path}")

    crops = []
    marker_points = []

    print("label_frame,node_id,t,z,y,x")
    for label_frame, (node_id, t, z, y, x) in enumerate(points):
        frame = np.asarray(image[t, z, :, :])
        if args.crop_size == 0:
            crops.append(frame)
            marker_points.append([label_frame, y, x])
        else:
            center = args.crop_size // 2
            crops.append(crop_2d(frame, y, x, args.crop_size))
            marker_points.append([label_frame, center, center])
        print(f"{label_frame},{node_id},{t},{z},{y},{x}")

    crop_stack = np.stack(crops, axis=0)
    marker_points = np.asarray(marker_points, dtype=float)

    viewer = napari.Viewer()
    try:
        viewer.dims.axis_labels = ("label_frame", "crop_y", "crop_x")
    except Exception:
        pass

    image_kwargs = {"name": f"{args.zarr_path.stem} label-centered crops"}
    if args.contrast_min is not None or args.contrast_max is not None:
        image_kwargs["contrast_limits"] = (
            0.0 if args.contrast_min is None else args.contrast_min,
            float(crop_stack.max()) if args.contrast_max is None else args.contrast_max,
        )

    viewer.add_image(crop_stack, **image_kwargs)
    viewer.add_points(
        marker_points,
        name="label center",
        size=12,
        face_color="magenta",
        opacity=1.0,
    )
    napari.run()


if __name__ == "__main__":
    main()
