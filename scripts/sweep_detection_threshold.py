from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from classical_detect_track import load_estimated_number_of_nodes, parse_sigma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep classical detector thresholds against estimated node count.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--geff", type=Path, default=None)
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/threshold_sweep.csv"))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--quantiles", default="99.5,99.3,99.1,99.0,98.9,98.7,98.5")
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--max-peaks-per-frame", type=int, default=3000)
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = zarr.open(args.zarr_path / args.array_path, mode="r")
    sigma_zyx = parse_sigma(args.gaussian_sigma_zyx)
    quantiles = [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
    frame_count = min(args.max_frames, int(image.shape[0]))
    estimated = load_estimated_number_of_nodes(args.geff) if args.geff else None

    counts = {q: 0 for q in quantiles}
    lowest_quantile = min(quantiles)
    for t in range(frame_count):
        volume = np.asarray(image[t, :, :, :])
        smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma_zyx)
        positive = smoothed[smoothed > 0]
        print(f"t={t:03d}: positive_voxels={len(positive)}")
        if len(positive) == 0:
            continue

        thresholds = {q: float(np.percentile(positive, q)) for q in quantiles}
        lowest_threshold = thresholds[lowest_quantile]
        peaks = peak_local_max(
            smoothed,
            min_distance=args.min_peak_distance,
            threshold_abs=lowest_threshold,
            exclude_border=False,
            num_peaks=args.max_peaks_per_frame,
        )
        if len(peaks) == 0:
            continue
        intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
        for q in quantiles:
            counts[q] += min(int(np.count_nonzero(intensities >= thresholds[q])), args.max_peaks_per_frame)

    rows = []
    for q in quantiles:
        count = counts[q]
        rows.append(
            {
                "threshold_quantile": q,
                "pred_nodes": count,
                "estimated_number_of_nodes": estimated,
                "pred_to_estimated_node_ratio": count / estimated if estimated else None,
                "estimated_minus_pred_nodes": estimated - count if estimated else None,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
