import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SUMMARY_FIELDS = [
    "dataset",
    "embryo",
    "seconds",
    "pred_nodes",
    "estimated_number_of_nodes",
    "pred_to_estimated_node_ratio",
    "gt_sparse_nodes",
    "matched_gt_nodes",
    "sparse_node_recall",
    "mean_matched_distance_um",
    "median_matched_distance_um",
    "pred_edges",
    "gt_sparse_edges",
    "sparse_edge_tp",
    "sparse_edge_recall",
    "labeled_subset_edge_precision",
    "pred_division_nodes",
    "gt_sparse_division_nodes",
    "sparse_division_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical detector/linker across all train GEFF pairs.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/batch_train_global_flow"))
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--threshold-quantile", type=float, default=50.0)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--max-peaks-per-frame", type=int, default=300)
    parser.add_argument("--contrast-mode", choices=["raw", "subtract", "ratio"], default="raw")
    parser.add_argument("--background-sigma-zyx", default="3.0,8.0,8.0")
    parser.add_argument("--contrast-ratio-epsilon", type=float, default=50.0)
    parser.add_argument("--enable-adaptive-detection", action="store_true")
    parser.add_argument("--adaptive-prescan-frames", type=int, default=5)
    parser.add_argument("--adaptive-min-peaks-per-frame", type=int, default=180)
    parser.add_argument("--adaptive-max-peaks-per-frame", type=int, default=850)
    parser.add_argument("--adaptive-mean-slope", type=float, default=0.65)
    parser.add_argument("--adaptive-mean-intercept", type=float, default=130.0)
    parser.add_argument("--adaptive-high-background-p50", type=float, default=1200.0)
    parser.add_argument("--adaptive-high-background-cap", type=int, default=330)
    parser.add_argument("--link-max-distance-um", type=float, default=7.0)
    parser.add_argument("--flow-confident-distance-um", type=float, default=4.0)
    parser.add_argument("--no-global-flow", action="store_true")
    parser.add_argument("--enable-blob-size-filter", action="store_true")
    parser.add_argument("--blob-size-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--blob-min-voxels", type=int, default=450)
    parser.add_argument("--blob-max-voxels", type=int, default=1400)
    parser.add_argument("--blob-target-voxels", type=float, default=928.0)
    parser.add_argument("--blob-size-sigma-voxels", type=float, default=350.0)
    parser.add_argument("--blob-size-penalty-weight", type=float, default=1.0)
    parser.add_argument("--blob-filter-oversample-factor", type=float, default=3.0)
    parser.add_argument("--blob-alpha", type=float, default=0.35)
    parser.add_argument("--blob-background-percentile", type=float, default=20.0)
    parser.add_argument("--blob-crop-radius-zyx", default="4,8,8")
    parser.add_argument("--enable-centroid-refinement", action="store_true")
    parser.add_argument("--centroid-alpha", type=float, default=0.25)
    parser.add_argument("--centroid-background-percentile", type=float, default=20.0)
    parser.add_argument("--centroid-crop-radius-zyx", default="5,10,10")
    parser.add_argument("--proposal-scorer-path", type=Path, default=None)
    parser.add_argument("--proposal-scorer-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-scorer-oversample-factor", type=float, default=3.0)
    parser.add_argument("--proposal-scorer-batch-size", type=int, default=512)
    parser.add_argument("--proposal-scorer-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--proposal-scorer-min-keep-fraction", type=float, default=0.5)
    parser.add_argument("--sample", action="append", default=None, help="Sample stem to include. Repeatable.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def find_pairs(train_dir: Path) -> list[tuple[str, Path, Path]]:
    zarrs = {p.name.removesuffix(".zarr"): p for p in train_dir.glob("*.zarr") if p.is_dir()}
    geffs = {p.name.removesuffix(".geff"): p for p in train_dir.glob("*.geff") if p.is_dir()}
    names = sorted(set(zarrs) & set(geffs))
    return [(name, zarrs[name], geffs[name]) for name in names]


def read_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {field: data.get(field) for field in SUMMARY_FIELDS if field not in {"dataset", "embryo", "seconds"}}


def write_summary(rows: list[dict], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = find_pairs(args.train_dir)
    if args.sample:
        wanted = set(args.sample)
        pairs = [pair for pair in pairs if pair[0] in wanted]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"No .zarr/.geff pairs found in {args.train_dir}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or (args.out_root / "summary.csv")
    rows: list[dict] = []

    for index, (name, zarr_path, geff_path) in enumerate(pairs, start=1):
        out_dir = args.out_root / name
        metrics_path = out_dir / "metrics.json"
        start = time.perf_counter()

        if args.skip_existing and metrics_path.exists():
            print(f"[{index}/{len(pairs)}] skipping existing {name}", flush=True)
            seconds = None
        else:
            print(f"[{index}/{len(pairs)}] processing {name}", flush=True)
            cmd = [
                sys.executable,
                "scripts/classical_detect_track.py",
                str(zarr_path),
                "--geff",
                str(geff_path),
                "--out-dir",
                str(out_dir),
                "--max-frames",
                str(args.max_frames),
                "--threshold-quantile",
                str(args.threshold_quantile),
                "--min-peak-distance",
                str(args.min_peak_distance),
                "--max-peaks-per-frame",
                str(args.max_peaks_per_frame),
                "--contrast-mode",
                args.contrast_mode,
                "--background-sigma-zyx",
                args.background_sigma_zyx,
                "--contrast-ratio-epsilon",
                str(args.contrast_ratio_epsilon),
                "--link-max-distance-um",
                str(args.link_max_distance_um),
                "--flow-confident-distance-um",
                str(args.flow_confident_distance_um),
            ]
            if not args.no_global_flow:
                cmd.append("--enable-global-flow")
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
                        "--blob-min-voxels",
                        str(args.blob_min_voxels),
                        "--blob-max-voxels",
                        str(args.blob_max_voxels),
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
            subprocess.run(cmd, check=True)
            seconds = time.perf_counter() - start

        metrics = read_metrics(metrics_path)
        row = {
            "dataset": name,
            "embryo": name.split("_")[0],
            "seconds": f"{seconds:.1f}" if seconds is not None else "",
            **metrics,
        }
        rows.append(row)
        write_summary(rows, summary_csv)
        print(
            f"  node_recall={row['sparse_node_recall']:.3f} "
            f"edge_recall={row['sparse_edge_recall']:.3f} "
            f"ratio={row['pred_to_estimated_node_ratio']:.3f}",
            flush=True,
        )

    write_summary(rows, summary_csv)
    print(f"wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
