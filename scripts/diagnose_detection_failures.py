from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


SCALE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize sparse-label detection misses for one sample.")
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--geff", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-match-distance-um", type=float, default=7.0)
    parser.add_argument("--crop-radius-yx", type=int, default=28)
    parser.add_argument("--crop-radius-z", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=12)
    return parser.parse_args()


def load_geff_nodes(geff_path: Path) -> np.ndarray:
    root = zarr.open_group(geff_path, mode="r")
    node_ids = np.asarray(root["nodes/ids"]).astype(np.int64)
    t = np.asarray(root["nodes/props/t/values"]).astype(np.int64)
    z = np.asarray(root["nodes/props/z/values"]).astype(np.int64)
    y = np.asarray(root["nodes/props/y/values"]).astype(np.int64)
    x = np.asarray(root["nodes/props/x/values"]).astype(np.int64)
    return np.stack([node_ids, t, z, y, x], axis=1)


def load_pred_nodes(nodes_csv: Path) -> list[dict]:
    with nodes_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": int(row["z"]),
            "y": int(row["y"]),
            "x": int(row["x"]),
            "score": float(row["score"]),
        }
        for row in rows
    ]


def match_and_summarize(gt_nodes: np.ndarray, pred_nodes: list[dict], max_distance_um: float) -> list[dict]:
    pred_by_t: dict[int, list[dict]] = {}
    gt_by_t: dict[int, list[np.ndarray]] = {}
    for pred in pred_nodes:
        pred_by_t.setdefault(pred["t"], []).append(pred)
    for gt in gt_nodes:
        gt_by_t.setdefault(int(gt[1]), []).append(gt)

    rows: list[dict] = []
    for t in sorted(gt_by_t):
        gts = np.asarray(gt_by_t[t], dtype=np.int64)
        preds = pred_by_t.get(t, [])
        gt_xyz = gts[:, 2:5].astype(np.float32) * SCALE_ZYX
        pred_xyz = np.asarray([[p["z"], p["y"], p["x"]] for p in preds], dtype=np.float32) * SCALE_ZYX
        matched_gt_indices: set[int] = set()
        pred_for_gt: dict[int, tuple[dict, float]] = {}

        if len(preds):
            dist = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=2)
            cost = dist.copy()
            cost[cost > max_distance_um] = 1e6
            gt_ind, pred_ind = linear_sum_assignment(cost)
            for gi, pi in zip(gt_ind, pred_ind):
                d = float(dist[gi, pi])
                if d <= max_distance_um:
                    matched_gt_indices.add(int(gi))
                    pred_for_gt[int(gi)] = (preds[int(pi)], d)

        for gi, gt in enumerate(gts):
            gt_coord_um = gt[2:5].astype(np.float32) * SCALE_ZYX
            if len(preds):
                distances = np.linalg.norm(pred_xyz - gt_coord_um, axis=1)
                nearest_index = int(np.argmin(distances))
                nearest = preds[nearest_index]
                nearest_distance = float(distances[nearest_index])
            else:
                nearest = None
                nearest_distance = float("inf")
            match = pred_for_gt.get(gi)
            rows.append(
                {
                    "gt_id": int(gt[0]),
                    "t": int(gt[1]),
                    "z": int(gt[2]),
                    "y": int(gt[3]),
                    "x": int(gt[4]),
                    "matched": gi in matched_gt_indices,
                    "matched_pred_id": match[0]["node_id"] if match else "",
                    "matched_distance_um": match[1] if match else "",
                    "nearest_pred_id": nearest["node_id"] if nearest else "",
                    "nearest_z": nearest["z"] if nearest else "",
                    "nearest_y": nearest["y"] if nearest else "",
                    "nearest_x": nearest["x"] if nearest else "",
                    "nearest_score": nearest["score"] if nearest else "",
                    "nearest_distance_um": nearest_distance,
                }
            )
    return rows


def robust_limits(image: np.ndarray) -> tuple[float, float]:
    positive = image[image > 0]
    if len(positive) == 0:
        return 0.0, 1.0
    return float(np.percentile(positive, 1)), float(np.percentile(positive, 99.7))


def crop_bounds(center: int, radius: int, limit: int) -> tuple[int, int]:
    lo = max(0, center - radius)
    hi = min(limit, center + radius + 1)
    return lo, hi


def normalize_plane(plane: np.ndarray) -> Image.Image:
    vmin, vmax = robust_limits(plane)
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((plane.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L").convert("RGB")


def draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, color: str, kind: str) -> None:
    if kind == "cross":
        draw.line((x - 6, y, x + 6, y), fill=color, width=2)
        draw.line((x, y - 6, x, y + 6), fill=color, width=2)
    else:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)


def render_case_panel(image, row: dict, crop_radius_yx: int, projection: bool) -> Image.Image:
    z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
    y0, y1 = crop_bounds(y, crop_radius_yx, image.shape[1])
    x0, x1 = crop_bounds(x, crop_radius_yx, image.shape[2])
    if projection:
        z0, z1 = crop_bounds(z, 3, image.shape[0])
        plane = image[z0:z1, y0:y1, x0:x1].max(axis=0)
    else:
        plane = image[z, y0:y1, x0:x1]
    panel = normalize_plane(plane).resize((160, 160), resample=Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    sx = 160 / max(1, x1 - x0)
    sy = 160 / max(1, y1 - y0)
    draw_marker(draw, int((x - x0) * sx), int((y - y0) * sy), "lime", "cross")
    if row["nearest_z"] != "":
        draw_marker(
            draw,
            int((int(row["nearest_x"]) - x0) * sx),
            int((int(row["nearest_y"]) - y0) * sy),
            "red",
            "circle",
        )
    return panel


def save_montage(
    image_array,
    rows: list[dict],
    out_path: Path,
    heading: str,
    crop_radius_yx: int,
    projection: bool,
) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    panel_w, panel_h = 160, 160
    label_h = 42
    gutter = 12
    heading_h = 32
    row_h = panel_h + label_h + gutter
    canvas = Image.new("RGB", (panel_w * 2 + gutter, heading_h + row_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 8), heading, fill="black", font=font)
    y_cursor = heading_h
    for row in rows:
        image = np.asarray(image_array[int(row["t"]), :, :, :])
        label = (
            f"gt {row['gt_id']} t={row['t']} zyx=({row['z']},{row['y']},{row['x']})\n"
            f"nearest d={float(row['nearest_distance_um']):.2f} um"
        )
        panel_slice = render_case_panel(image, row, crop_radius_yx, projection=False)
        panel_proj = render_case_panel(image, row, crop_radius_yx, projection=True)
        canvas.paste(panel_slice, (0, y_cursor + label_h))
        canvas.paste(panel_proj, (panel_w + gutter, y_cursor + label_h))
        draw.text((4, y_cursor), label, fill="black", font=font)
        draw.text((panel_w + gutter + 4, y_cursor), "max Z crop", fill="black", font=font)
        y_cursor += row_h
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = zarr.open(args.zarr / "0", mode="r")
    gt_nodes = load_geff_nodes(args.geff)
    pred_nodes = load_pred_nodes(args.nodes_csv)
    rows = match_and_summarize(gt_nodes, pred_nodes, args.max_match_distance_um)

    rows_path = args.out_dir / "gt_nearest_prediction_cases.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    misses = [row for row in rows if not row["matched"]]
    matches = [row for row in rows if row["matched"]]
    misses_sorted = sorted(misses, key=lambda row: float(row["nearest_distance_um"]), reverse=True)
    border_misses = sorted(misses, key=lambda row: abs(float(row["nearest_distance_um"]) - args.max_match_distance_um))
    matches_sorted = sorted(matches, key=lambda row: float(row["matched_distance_um"] or 0), reverse=True)

    save_montage(
        image,
        misses_sorted[: args.max_cases],
        args.out_dir / "worst_misses.png",
        "Worst sparse-label misses: green=GT, red=nearest predicted node",
        args.crop_radius_yx,
        projection=False,
    )
    save_montage(
        image,
        border_misses[: args.max_cases],
        args.out_dir / "near_threshold_misses.png",
        "Near-threshold sparse-label misses: green=GT, red=nearest predicted node",
        args.crop_radius_yx,
        projection=False,
    )
    save_montage(
        image,
        matches_sorted[: args.max_cases],
        args.out_dir / "worst_matches.png",
        "Weakest accepted sparse-label matches: green=GT, red=matched/nearest predicted node",
        args.crop_radius_yx,
        projection=False,
    )

    print(f"gt labels: {len(rows)}")
    print(f"matched: {len(matches)}")
    print(f"missed: {len(misses)}")
    print(f"wrote {rows_path}")
    print(f"wrote {args.out_dir / 'worst_misses.png'}")
    print(f"wrote {args.out_dir / 'near_threshold_misses.png'}")
    print(f"wrote {args.out_dir / 'worst_matches.png'}")


if __name__ == "__main__":
    main()
