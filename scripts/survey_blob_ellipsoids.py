from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import gaussian_filter, label
from skimage.feature import peak_local_max


SCALE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def parse_sigma(text: str) -> tuple[float, float, float]:
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey approximate blob/ellipsoid sizes around classical 3D peak detections."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--sample", action="append", default=None, help="Sample stem to include. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ellipsoid_survey"))
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--max-frames-per-sample", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--threshold-quantile", type=float, default=50.0)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--max-peaks-per-frame", type=int, default=150)
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")
    parser.add_argument("--crop-radius-zyx", default="4,8,8")
    parser.add_argument(
        "--blob-alpha",
        type=float,
        default=0.35,
        help="Local component threshold: background + alpha * (peak - background).",
    )
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=20.0,
        help="Local background percentile inside each crop.",
    )
    parser.add_argument("--min-component-voxels", type=int, default=3)
    return parser.parse_args()


def find_zarr_samples(train_dir: Path, requested: list[str] | None) -> list[tuple[str, Path]]:
    zarrs = sorted(p for p in train_dir.glob("*.zarr") if p.is_dir())
    if requested:
        wanted = set(requested)
        zarrs = [p for p in zarrs if p.name.removesuffix(".zarr") in wanted]
    return [(p.name.removesuffix(".zarr"), p) for p in zarrs]


def detect_peaks(
    volume: np.ndarray,
    sigma_zyx: tuple[float, float, float],
    threshold_quantile: float,
    min_peak_distance: int,
    max_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma_zyx)
    positive = smoothed[smoothed > 0]
    if len(positive) == 0:
        return np.zeros((0, 3), dtype=np.int64), smoothed
    threshold = float(np.percentile(positive, threshold_quantile))
    peaks = peak_local_max(
        smoothed,
        min_distance=min_peak_distance,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=max_peaks,
    )
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.int64), smoothed
    intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    order = np.argsort(-intensities)
    return peaks[order], smoothed


def crop_bounds(center: np.ndarray, radius: np.ndarray, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, np.asarray(shape))
    return lo.astype(int), hi.astype(int)


def component_for_peak(
    smoothed: np.ndarray,
    peak_zyx: np.ndarray,
    crop_radius_zyx: np.ndarray,
    alpha: float,
    background_percentile: float,
    min_component_voxels: int,
) -> dict | None:
    lo, hi = crop_bounds(peak_zyx, crop_radius_zyx, smoothed.shape)
    crop = smoothed[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    local_peak = peak_zyx - lo
    peak_value = float(crop[tuple(local_peak)])
    background = float(np.percentile(crop, background_percentile))
    threshold = background + alpha * (peak_value - background)
    mask = crop >= threshold
    labels, count = label(mask)
    if count == 0:
        return None
    component_id = int(labels[tuple(local_peak)])
    if component_id == 0:
        return None
    coords_local = np.argwhere(labels == component_id)
    voxel_count = int(len(coords_local))
    if voxel_count < min_component_voxels:
        return None

    coords_global = coords_local + lo
    coords_um = coords_global.astype(np.float32) * SCALE_ZYX
    weights = crop[coords_local[:, 0], coords_local[:, 1], coords_local[:, 2]].astype(np.float64)
    weights = np.maximum(weights - background, 1e-6)
    centroid_um = np.average(coords_um, axis=0, weights=weights)
    centered = coords_um - centroid_um
    covariance = (centered * weights[:, None]).T @ centered / float(weights.sum())
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.maximum(np.sort(eigvals)[::-1], 0.0)

    extents_vox = coords_global.max(axis=0) - coords_global.min(axis=0) + 1
    extents_um = extents_vox.astype(np.float32) * SCALE_ZYX
    voxel_volume_um3 = float(np.prod(SCALE_ZYX))
    volume_um3 = voxel_count * voxel_volume_um3
    equivalent_radius_um = float((3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    rms_axes_um = np.sqrt(eigvals)

    return {
        "voxel_count": voxel_count,
        "volume_um3": volume_um3,
        "equivalent_radius_um": equivalent_radius_um,
        "extent_z_um": float(extents_um[0]),
        "extent_y_um": float(extents_um[1]),
        "extent_x_um": float(extents_um[2]),
        "rms_axis1_um": float(rms_axes_um[0]),
        "rms_axis2_um": float(rms_axes_um[1]),
        "rms_axis3_um": float(rms_axes_um[2]),
        "peak_value": peak_value,
        "background": background,
        "threshold": float(threshold),
        "centroid_z_um": float(centroid_um[0]),
        "centroid_y_um": float(centroid_um[1]),
        "centroid_x_um": float(centroid_um[2]),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p25": None, "median": None, "p75": None, "p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    args = parse_args()
    sigma_zyx = parse_sigma(args.gaussian_sigma_zyx)
    crop_radius_zyx = np.asarray(parse_sigma(args.crop_radius_zyx), dtype=int)
    samples = find_zarr_samples(args.train_dir, args.sample)
    if not samples:
        raise SystemExit(f"No samples found in {args.train_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    start = time.perf_counter()

    for sample_index, (sample_name, zarr_path) in enumerate(samples, start=1):
        image = zarr.open(zarr_path / args.array_path, mode="r")
        frame_indices = list(range(0, int(image.shape[0]), args.frame_stride))[: args.max_frames_per_sample]
        print(f"[{sample_index}/{len(samples)}] {sample_name}: {len(frame_indices)} frames", flush=True)
        for t in frame_indices:
            volume = np.asarray(image[t, :, :, :])
            peaks, smoothed = detect_peaks(
                volume,
                sigma_zyx=sigma_zyx,
                threshold_quantile=args.threshold_quantile,
                min_peak_distance=args.min_peak_distance,
                max_peaks=args.max_peaks_per_frame,
            )
            for rank, peak in enumerate(peaks, start=1):
                component = component_for_peak(
                    smoothed,
                    peak.astype(int),
                    crop_radius_zyx=crop_radius_zyx,
                    alpha=args.blob_alpha,
                    background_percentile=args.background_percentile,
                    min_component_voxels=args.min_component_voxels,
                )
                if component is None:
                    continue
                rows.append(
                    {
                        "dataset": sample_name,
                        "embryo": sample_name.split("_")[0],
                        "t": int(t),
                        "rank": rank,
                        "peak_z": int(peak[0]),
                        "peak_y": int(peak[1]),
                        "peak_x": int(peak[2]),
                        **component,
                    }
                )

    fieldnames = list(rows[0].keys()) if rows else ["dataset"]
    detail_csv = args.out_dir / "blob_ellipsoid_details.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(samples),
        "blobs": len(rows),
        "seconds": time.perf_counter() - start,
        "settings": {
            "max_frames_per_sample": args.max_frames_per_sample,
            "frame_stride": args.frame_stride,
            "threshold_quantile": args.threshold_quantile,
            "min_peak_distance": args.min_peak_distance,
            "max_peaks_per_frame": args.max_peaks_per_frame,
            "gaussian_sigma_zyx": args.gaussian_sigma_zyx,
            "crop_radius_zyx": args.crop_radius_zyx,
            "blob_alpha": args.blob_alpha,
            "background_percentile": args.background_percentile,
        },
        "overall": {
            "voxel_count": summarize([r["voxel_count"] for r in rows]),
            "volume_um3": summarize([r["volume_um3"] for r in rows]),
            "equivalent_radius_um": summarize([r["equivalent_radius_um"] for r in rows]),
            "extent_z_um": summarize([r["extent_z_um"] for r in rows]),
            "extent_y_um": summarize([r["extent_y_um"] for r in rows]),
            "extent_x_um": summarize([r["extent_x_um"] for r in rows]),
            "rms_axis1_um": summarize([r["rms_axis1_um"] for r in rows]),
            "rms_axis2_um": summarize([r["rms_axis2_um"] for r in rows]),
            "rms_axis3_um": summarize([r["rms_axis3_um"] for r in rows]),
        },
        "by_embryo": {},
    }
    for embryo in sorted({r["embryo"] for r in rows}):
        embryo_rows = [r for r in rows if r["embryo"] == embryo]
        summary["by_embryo"][embryo] = {
            "blobs": len(embryo_rows),
            "volume_um3": summarize([r["volume_um3"] for r in embryo_rows]),
            "equivalent_radius_um": summarize([r["equivalent_radius_um"] for r in embryo_rows]),
            "voxel_count": summarize([r["voxel_count"] for r in embryo_rows]),
        }

    summary_json = args.out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print(f"wrote {detail_csv}", flush=True)
    print(f"wrote {summary_json}", flush=True)


if __name__ == "__main__":
    main()
