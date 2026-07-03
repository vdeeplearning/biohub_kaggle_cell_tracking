import argparse
import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
from skimage.feature import peak_local_max


def parse_sigma(text: str) -> tuple[float, float, float]:
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--gaussian-sigma-zyx must contain three comma-separated values")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile detection substeps for a Biohub Zarr sample.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--threshold-quantile", type=float, default=50.0)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--max-peaks-per-frame", type=int, default=300)
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")
    parser.add_argument("--smooth-method", choices=["gaussian", "box2", "box3", "none"], default="gaussian")
    parser.add_argument("--box-size-zyx", default="3,3,3")
    parser.add_argument("--peak-method", choices=["skimage", "max-filter"], default="skimage")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigma = parse_sigma(args.gaussian_sigma_zyx)
    box_size = tuple(int(round(v)) for v in parse_sigma(args.box_size_zyx))
    image = zarr.open(args.zarr_path / args.array_path, mode="r")
    frame_count = min(args.max_frames, int(image.shape[0]))

    totals = {
        "read_seconds": 0.0,
        "astype_seconds": 0.0,
        "gaussian_seconds": 0.0,
        "positive_filter_seconds": 0.0,
        "percentile_seconds": 0.0,
        "peak_local_max_seconds": 0.0,
        "sort_pack_seconds": 0.0,
        "total_seconds": 0.0,
        "frames": frame_count,
        "peaks": 0,
    }

    for t in range(frame_count):
        frame_start = time.perf_counter()

        start = time.perf_counter()
        volume = np.asarray(image[t, :, :, :])
        totals["read_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        volume_f = volume.astype(np.float32)
        totals["astype_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        if args.smooth_method == "gaussian":
            smoothed = gaussian_filter(volume_f, sigma=sigma)
        elif args.smooth_method == "box2":
            smoothed = uniform_filter(volume_f, size=box_size)
            smoothed = uniform_filter(smoothed, size=box_size)
        elif args.smooth_method == "box3":
            smoothed = uniform_filter(volume_f, size=box_size)
            smoothed = uniform_filter(smoothed, size=box_size)
            smoothed = uniform_filter(smoothed, size=box_size)
        else:
            smoothed = volume_f
        totals["gaussian_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        positive = smoothed[smoothed > 0]
        totals["positive_filter_seconds"] += time.perf_counter() - start

        if len(positive) == 0:
            totals["total_seconds"] += time.perf_counter() - frame_start
            continue

        start = time.perf_counter()
        threshold = float(np.percentile(positive, args.threshold_quantile))
        totals["percentile_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        if args.peak_method == "skimage":
            peaks = peak_local_max(
                smoothed,
                min_distance=args.min_peak_distance,
                threshold_abs=threshold,
                exclude_border=False,
                num_peaks=args.max_peaks_per_frame,
            )
        else:
            size = 2 * args.min_peak_distance + 1
            local_max = maximum_filter(smoothed, size=size, mode="nearest")
            peaks = np.argwhere((smoothed == local_max) & (smoothed > threshold))
        totals["peak_local_max_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        if len(peaks):
            intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
            order = np.argsort(-intensities)
            if args.max_peaks_per_frame > 0:
                order = order[: args.max_peaks_per_frame]
            peaks = peaks[order]
        totals["sort_pack_seconds"] += time.perf_counter() - start

        totals["peaks"] += int(len(peaks))
        totals["total_seconds"] += time.perf_counter() - frame_start

    timed_keys = [key for key in totals if key.endswith("_seconds") and key != "total_seconds"]
    totals["accounted_seconds"] = sum(float(totals[key]) for key in timed_keys)
    totals["other_seconds"] = float(totals["total_seconds"]) - float(totals["accounted_seconds"])
    totals["seconds_per_frame"] = float(totals["total_seconds"]) / max(1, frame_count)

    print(json.dumps(totals, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(totals, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
