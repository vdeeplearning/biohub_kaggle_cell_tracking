from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr


VOXEL_SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View sparse GEFF cell division labels overlaid on a Biohub Zarr volume."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to one .zarr sample.")
    parser.add_argument("geff_path", type=Path, help="Path to the matching .geff sample.")
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--contrast-min", type=float, default=None)
    parser.add_argument("--contrast-max", type=float, default=None)
    parser.add_argument(
        "--division-index",
        type=int,
        default=0,
        help="Which labeled division to jump to initially, 0-based in the printed list.",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument(
        "--time-radius",
        type=int,
        default=3,
        help="Number of frames before/after the division parent time to show in the cropped viewer.",
    )
    parser.add_argument(
        "--crop-radius-zyx",
        default="8,48,48",
        help="Spatial crop radius around the selected division, as z,y,x voxels.",
    )
    parser.add_argument(
        "--plane",
        choices=["auto", "xy", "xz", "yz", "volume"],
        default="auto",
        help=(
            "Viewing plane. auto chooses the orthogonal projection where daughters are most separated; "
            "volume shows the cropped 4D volume."
        ),
    )
    parser.add_argument(
        "--only-division-index",
        action="store_true",
        help="Show only the selected division instead of all labeled divisions in the GEFF.",
    )
    return parser.parse_args()


def load_geff(geff_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(geff_path, mode="r")
    node_ids = np.asarray(root["nodes/ids"]).astype(np.int64)
    t = np.asarray(root["nodes/props/t/values"]).astype(np.int64)
    z = np.asarray(root["nodes/props/z/values"]).astype(np.int64)
    y = np.asarray(root["nodes/props/y/values"]).astype(np.int64)
    x = np.asarray(root["nodes/props/x/values"]).astype(np.int64)
    nodes = np.stack([node_ids, t, z, y, x], axis=1)
    edges = np.asarray(root["edges/ids"]).astype(np.int64)
    if edges.ndim == 1 and len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    return nodes, edges


def find_divisions(nodes: np.ndarray, edges: np.ndarray) -> list[dict]:
    by_id = {int(row[0]): row for row in nodes}
    targets_by_source: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in edges:
        targets_by_source[int(source_id)].append(int(target_id))

    divisions = []
    for source_id, target_ids in sorted(targets_by_source.items()):
        if len(target_ids) < 2 or source_id not in by_id:
            continue
        valid_targets = [target_id for target_id in target_ids if target_id in by_id]
        if len(valid_targets) < 2:
            continue

        parent = by_id[source_id]
        daughter_rows = sorted([by_id[target_id] for target_id in valid_targets], key=lambda row: int(row[0]))
        daughter_rows = daughter_rows[:2]
        divisions.append(
            {
                "parent": parent,
                "daughters": daughter_rows,
                "source_id": source_id,
                "target_ids": [int(row[0]) for row in daughter_rows],
            }
        )
    return divisions


def as_point(row: np.ndarray) -> list[float]:
    return [float(row[1]), float(row[2]), float(row[3]), float(row[4])]


def parse_int_triplet(text: str) -> tuple[int, int, int]:
    parts = [int(round(float(p.strip()))) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--crop-radius-zyx must have 3 comma-separated values")
    return tuple(parts)


def crop_bounds_for_division(division: dict, image_shape: tuple[int, ...], time_radius: int, crop_radius_zyx: tuple[int, int, int]) -> tuple[np.ndarray, tuple[slice, slice, slice, slice]]:
    rows = [division["parent"], *division["daughters"]]
    coords = np.asarray([row[1:5] for row in rows], dtype=int)
    center = np.rint(coords.mean(axis=0)).astype(int)
    radius = np.asarray([time_radius, *crop_radius_zyx], dtype=int)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, np.asarray(image_shape, dtype=int))

    # Always include the parent and daughter coordinates, even when their midpoint is off-center.
    lo = np.minimum(lo, coords.min(axis=0))
    hi = np.maximum(hi, coords.max(axis=0) + 1)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(image_shape, dtype=int))
    return lo, tuple(slice(int(lo[i]), int(hi[i])) for i in range(4))


def choose_best_plane(division: dict) -> str:
    daughters = np.asarray([row[2:5] for row in division["daughters"]], dtype=float)
    delta_um = np.abs(daughters[1] - daughters[0]) * np.asarray(VOXEL_SCALE_TZYX[1:], dtype=float)
    drop_axis = int(np.argmin(delta_um))
    return ["yz", "xz", "xy"][drop_axis]


def project_image(cropped_image: da.Array, plane: str) -> tuple[da.Array, tuple[float, float, float], tuple[str, str, str]]:
    if plane == "xy":
        return cropped_image.max(axis=1), (VOXEL_SCALE_TZYX[0], VOXEL_SCALE_TZYX[2], VOXEL_SCALE_TZYX[3]), ("t", "y", "x")
    if plane == "xz":
        return cropped_image.max(axis=2), (VOXEL_SCALE_TZYX[0], VOXEL_SCALE_TZYX[1], VOXEL_SCALE_TZYX[3]), ("t", "z", "x")
    if plane == "yz":
        return cropped_image.max(axis=3), (VOXEL_SCALE_TZYX[0], VOXEL_SCALE_TZYX[1], VOXEL_SCALE_TZYX[2]), ("t", "z", "y")
    raise ValueError(f"Unsupported plane: {plane}")


def project_points(points: list[list[float]], plane: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if plane == "xy":
        return array[:, [0, 2, 3]]
    if plane == "xz":
        return array[:, [0, 1, 3]]
    if plane == "yz":
        return array[:, [0, 1, 2]]
    raise ValueError(f"Unsupported plane: {plane}")


def project_lines(line_data: list[np.ndarray], plane: str) -> list[np.ndarray]:
    return [project_points(line.tolist(), plane) for line in line_data]


def main() -> None:
    args = parse_args()
    image = da.from_zarr(zarr.open(args.zarr_path / args.array_path, mode="r"))
    nodes, edges = load_geff(args.geff_path)
    divisions = find_divisions(nodes, edges)

    print(f"Opening {args.zarr_path}")
    print(f"shape={image.shape}, dtype={image.dtype}, chunks={image.chunksize}")
    print(f"Found {len(divisions)} labeled division(s) in {args.geff_path}")
    print("division_index,parent_id,parent_t,parent_z,parent_y,parent_x,daughter_a_id,daughter_a_t,daughter_a_z,daughter_a_y,daughter_a_x,daughter_b_id,daughter_b_t,daughter_b_z,daughter_b_y,daughter_b_x")
    for i, division in enumerate(divisions):
        p = division["parent"]
        a, b = division["daughters"]
        print(
            f"{i},{int(p[0])},{int(p[1])},{int(p[2])},{int(p[3])},{int(p[4])},"
            f"{int(a[0])},{int(a[1])},{int(a[2])},{int(a[3])},{int(a[4])},"
            f"{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])},{int(b[4])}"
        )

    if not divisions:
        raise ValueError("No labeled divisions found. A division is a GEFF source node with two outgoing edges.")
    if args.division_index < 0 or args.division_index >= len(divisions):
        raise ValueError(f"--division-index must be between 0 and {len(divisions) - 1}")

    selected_division = divisions[args.division_index]
    crop_radius_zyx = parse_int_triplet(args.crop_radius_zyx)
    crop_origin, crop_slices = crop_bounds_for_division(
        selected_division,
        tuple(int(v) for v in image.shape),
        args.time_radius,
        crop_radius_zyx,
    )
    cropped_image = image[crop_slices]
    plane = choose_best_plane(selected_division) if args.plane == "auto" else args.plane
    print(
        "Crop origin t,z,y,x="
        f"{crop_origin.tolist()} shape={tuple(int(v) for v in cropped_image.shape)} "
        f"slices={[(s.start, s.stop) for s in crop_slices]}"
    )
    print(f"Viewing plane: {plane}")

    if args.only_division_index:
        visible_divisions = [selected_division]
    else:
        # In cropped mode, only show divisions whose parent is inside this crop.
        visible_divisions = [
            division
            for division in divisions
            if np.all(division["parent"][1:5] >= crop_origin)
            and np.all(division["parent"][1:5] < crop_origin + np.asarray(cropped_image.shape))
        ]

    parent_points = []
    daughter_points = []
    midpoint_points = []
    parent_labels = []
    daughter_labels = []
    line_data = []

    for i, division in enumerate(visible_divisions):
        original_index = args.division_index if args.only_division_index else i
        parent = division["parent"]
        daughters = division["daughters"]
        parent_point = (np.asarray(as_point(parent), dtype=float) - crop_origin).tolist()
        parent_points.append(parent_point)
        parent_labels.append(f"D{original_index} parent {int(parent[0])}")

        daughter_coords = []
        for daughter_number, daughter in enumerate(daughters, start=1):
            daughter_point = (np.asarray(as_point(daughter), dtype=float) - crop_origin).tolist()
            daughter_points.append(daughter_point)
            daughter_coords.append(daughter_point)
            daughter_labels.append(f"D{original_index} daughter {daughter_number} {int(daughter[0])}")
            line_data.append(np.asarray([parent_point, daughter_point], dtype=float))

        midpoint = np.mean(np.asarray(daughter_coords, dtype=float), axis=0)
        midpoint_points.append(midpoint.tolist())

    viewer = napari.Viewer()
    if plane == "volume":
        display_image = cropped_image
        display_scale = VOXEL_SCALE_TZYX
        axis_labels = ("t", "z", "y", "x")
        display_parent_points = np.asarray(parent_points, dtype=float)
        display_daughter_points = np.asarray(daughter_points, dtype=float)
        display_midpoint_points = np.asarray(midpoint_points, dtype=float)
        display_line_data = line_data
        text_translation = [0, 0, 4, 4]
    else:
        display_image, display_scale, axis_labels = project_image(cropped_image, plane)
        display_parent_points = project_points(parent_points, plane)
        display_daughter_points = project_points(daughter_points, plane)
        display_midpoint_points = project_points(midpoint_points, plane)
        display_line_data = project_lines(line_data, plane)
        text_translation = [0, 4, 4]
    try:
        viewer.dims.axis_labels = axis_labels
    except Exception:
        pass

    image_kwargs = {
        "name": f"{args.zarr_path.stem} division {plane}",
        "scale": display_scale,
        "blending": "additive",
    }
    if args.contrast_min is not None or args.contrast_max is not None:
        image_kwargs["contrast_limits"] = (
            0.0 if args.contrast_min is None else args.contrast_min,
            float(display_image.max().compute()) if args.contrast_max is None else args.contrast_max,
        )
    viewer.add_image(display_image, **image_kwargs)

    viewer.add_points(
        display_parent_points,
        name="division parents",
        scale=display_scale,
        size=args.point_size,
        face_color="yellow",
        opacity=1.0,
        text={"string": parent_labels, "size": 7, "color": "yellow", "translation": text_translation},
    )
    viewer.add_points(
        display_daughter_points,
        name="division daughters",
        scale=display_scale,
        size=args.point_size * 0.9,
        face_color="cyan",
        opacity=1.0,
        text={"string": daughter_labels, "size": 7, "color": "cyan", "translation": text_translation},
    )
    viewer.add_points(
        display_midpoint_points,
        name="daughter midpoints",
        scale=display_scale,
        size=args.point_size * 0.6,
        face_color="magenta",
        opacity=0.8,
    )
    if display_line_data:
        viewer.add_shapes(
            display_line_data,
            shape_type="line",
            name="parent-to-daughter edges",
            scale=display_scale,
            edge_color="lime",
            edge_width=2,
            opacity=0.9,
        )

    selected = divisions[args.division_index]["parent"].astype(int)
    selected_crop_step = (selected[1:5] - crop_origin).astype(int)
    if plane == "xy":
        selected_step = selected_crop_step[[0, 2, 3]]
    elif plane == "xz":
        selected_step = selected_crop_step[[0, 1, 3]]
    elif plane == "yz":
        selected_step = selected_crop_step[[0, 1, 2]]
    else:
        selected_step = selected_crop_step
    try:
        viewer.dims.current_step = tuple(selected_step.tolist())
    except Exception:
        pass
    print(
        f"Jumped to division {args.division_index}: "
        f"parent_id={int(selected[0])}, t={int(selected[1])}, z={int(selected[2])}, "
        f"y={int(selected[3])}, x={int(selected[4])}"
    )
    napari.run()


if __name__ == "__main__":
    main()
