from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical detection/linking with optional CNN proposal rescoring and write submission.csv."
    )
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/cnn_rescore_outputs"))
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--threshold-quantile", type=float, default=35.0)
    parser.add_argument("--min-peak-distance", type=int, default=2)
    parser.add_argument("--max-peaks-per-frame", type=int, default=850)
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")

    parser.add_argument("--enable-adaptive-detection", action="store_true", default=True)
    parser.add_argument("--adaptive-prescan-frames", type=int, default=5)
    parser.add_argument("--adaptive-min-peaks-per-frame", type=int, default=180)
    parser.add_argument("--adaptive-max-peaks-per-frame", type=int, default=850)
    parser.add_argument("--adaptive-mean-slope", type=float, default=0.65)
    parser.add_argument("--adaptive-mean-intercept", type=float, default=130.0)
    parser.add_argument("--adaptive-high-background-p50", type=float, default=1200.0)
    parser.add_argument("--adaptive-high-background-cap", type=int, default=330)

    parser.add_argument("--enable-blob-size-filter", action="store_true", default=True)
    parser.add_argument("--blob-size-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--blob-target-voxels", type=float, default=928.0)
    parser.add_argument("--blob-size-sigma-voxels", type=float, default=350.0)
    parser.add_argument("--blob-size-penalty-weight", type=float, default=1.0)
    parser.add_argument("--blob-filter-oversample-factor", type=float, default=3.0)
    parser.add_argument("--blob-alpha", type=float, default=0.35)
    parser.add_argument("--blob-background-percentile", type=float, default=20.0)
    parser.add_argument("--blob-crop-radius-zyx", default="4,8,8")

    parser.add_argument("--enable-centroid-refinement", action="store_true", default=True)
    parser.add_argument("--centroid-alpha", type=float, default=0.25)
    parser.add_argument("--centroid-background-percentile", type=float, default=20.0)
    parser.add_argument("--centroid-crop-radius-zyx", default="5,10,10")

    parser.add_argument("--proposal-scorer-path", type=Path, default=None)
    parser.add_argument("--proposal-scorer-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-scorer-oversample-factor", type=float, default=3.0)
    parser.add_argument("--proposal-scorer-batch-size", type=int, default=512)
    parser.add_argument("--proposal-scorer-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--proposal-scorer-min-keep-fraction", type=float, default=0.5)

    parser.add_argument("--link-max-distance-um", type=float, default=7.0)
    parser.add_argument("--enable-global-flow", action="store_true", default=True)
    parser.add_argument("--flow-confident-distance-um", type=float, default=4.0)
    parser.add_argument("--enable-divisions", action="store_true")
    return parser.parse_args()


def find_test_zarrs(input_root: Path, test_dir: Path | None) -> list[Path]:
    if test_dir is not None:
        return sorted(test_dir.glob("*.zarr"))

    known = input_root / "competitions" / "biohub-cell-tracking-during-development" / "test"
    if known.exists():
        return sorted(known.glob("*.zarr"))

    # Slow fallback, but useful outside Kaggle's usual competition mount layout.
    for candidate in input_root.glob("*/test"):
        zarrs = sorted(candidate.glob("*.zarr"))
        if zarrs:
            return zarrs
    return []


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_submission(result_dirs: list[Path], out_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for result_dir in result_dirs:
        for node in read_csv(result_dir / "nodes.csv"):
            rows.append(
                {
                    "dataset": node["dataset"],
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": int(node["z"]),
                    "y": int(node["y"]),
                    "x": int(node["x"]),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
        for edge in read_csv(result_dir / "edges.csv"):
            rows.append(
                {
                    "dataset": edge["dataset"],
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                }
            )

    for row_id, row in enumerate(rows):
        row["id"] = row_id

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    node_count = sum(1 for row in rows if row["row_type"] == "node")
    edge_count = sum(1 for row in rows if row["row_type"] == "edge")
    print(f"wrote {out_path}: nodes={node_count}, edges={edge_count}, rows={len(rows)}")


def append_common_detector_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--threshold-quantile",
            str(args.threshold_quantile),
            "--min-peak-distance",
            str(args.min_peak_distance),
            "--max-peaks-per-frame",
            str(args.max_peaks_per_frame),
            "--gaussian-sigma-zyx",
            args.gaussian_sigma_zyx,
            "--link-max-distance-um",
            str(args.link_max_distance_um),
        ]
    )
    if args.max_frames is not None:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.enable_adaptive_detection:
        cmd.extend(
            [
                "--enable-adaptive-detection",
                "--adaptive-prescan-frames",
                str(args.adaptive_prescan_frames),
                "--adaptive-min-peaks-per-frame",
                str(args.adaptive_min_peaks_per_frame),
                "--adaptive-max-peaks-per-frame",
                str(args.adaptive_max_peaks_per_frame),
                "--adaptive-mean-slope",
                str(args.adaptive_mean_slope),
                "--adaptive-mean-intercept",
                str(args.adaptive_mean_intercept),
                "--adaptive-high-background-p50",
                str(args.adaptive_high_background_p50),
                "--adaptive-high-background-cap",
                str(args.adaptive_high_background_cap),
            ]
        )
    if args.enable_blob_size_filter:
        cmd.extend(
            [
                "--enable-blob-size-filter",
                "--blob-size-mode",
                args.blob_size_mode,
                "--blob-target-voxels",
                str(args.blob_target_voxels),
                "--blob-size-sigma-voxels",
                str(args.blob_size_sigma_voxels),
                "--blob-size-penalty-weight",
                str(args.blob_size_penalty_weight),
                "--blob-filter-oversample-factor",
                str(args.blob_filter_oversample_factor),
                "--blob-alpha",
                str(args.blob_alpha),
                "--blob-background-percentile",
                str(args.blob_background_percentile),
                "--blob-crop-radius-zyx",
                args.blob_crop_radius_zyx,
            ]
        )
    if args.enable_centroid_refinement:
        cmd.extend(
            [
                "--enable-centroid-refinement",
                "--centroid-alpha",
                str(args.centroid_alpha),
                "--centroid-background-percentile",
                str(args.centroid_background_percentile),
                "--centroid-crop-radius-zyx",
                args.centroid_crop_radius_zyx,
            ]
        )
    if args.proposal_scorer_path is not None:
        cmd.extend(
            [
                "--proposal-scorer-path",
                str(args.proposal_scorer_path),
                "--proposal-scorer-threshold",
                str(args.proposal_scorer_threshold),
                "--proposal-scorer-oversample-factor",
                str(args.proposal_scorer_oversample_factor),
                "--proposal-scorer-batch-size",
                str(args.proposal_scorer_batch_size),
                "--proposal-scorer-device",
                args.proposal_scorer_device,
                "--proposal-scorer-min-keep-fraction",
                str(args.proposal_scorer_min_keep_fraction),
            ]
        )
    if args.enable_global_flow:
        cmd.extend(["--enable-global-flow", "--flow-confident-distance-um", str(args.flow_confident_distance_um)])
    if args.enable_divisions:
        cmd.append("--enable-divisions")


def main() -> None:
    args = parse_args()
    test_zarrs = find_test_zarrs(args.input_root, args.test_dir)
    print(f"found {len(test_zarrs)} test zarr samples")
    for path in test_zarrs[:10]:
        print(path)
    if not test_zarrs:
        raise FileNotFoundError("No test .zarr samples found.")

    result_dirs = []
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for index, zarr_path in enumerate(test_zarrs, start=1):
        out_dir = args.work_dir / zarr_path.stem
        result_dirs.append(out_dir)
        print(f"[{index}/{len(test_zarrs)}] processing {zarr_path.stem}", flush=True)
        cmd = [
            sys.executable,
            "scripts/classical_detect_track.py",
            str(zarr_path),
            "--out-dir",
            str(out_dir),
        ]
        append_common_detector_args(cmd, args)
        subprocess.run(cmd, check=True)

    write_submission(result_dirs, args.out)


if __name__ == "__main__":
    main()
