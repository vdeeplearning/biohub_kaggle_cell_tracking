import json
import tempfile
import py_compile
from pathlib import Path


SOURCE_NOTEBOOK = Path("biohub_classical_submission_fast.ipynb")
OUTPUT_NOTEBOOK = Path("biohub_classical_submission_adaptive_centroid_soft_ellipsoid.ipynb")


CONFIG_APPEND = """

# Global-flow linker.
ENABLE_GLOBAL_FLOW = True
FLOW_CONFIDENT_DISTANCE_UM = 4.0

# Soft ellipsoid/blob-size tuned ranking. These values come from the local blob survey.
ENABLE_BLOB_SIZE_RANKING = True
BLOB_TARGET_VOXELS = 928.0
BLOB_SIZE_SIGMA_VOXELS = 350.0
BLOB_SIZE_PENALTY_WEIGHT = 1.0
BLOB_RANKING_OVERSAMPLE_FACTOR = 3.0
BLOB_ALPHA = 0.35
BLOB_BACKGROUND_PERCENTILE = 20.0
BLOB_CROP_RADIUS_ZYX = (4, 8, 8)

# Adaptive per-sample detection cap from a cheap raw-intensity prescan.
ENABLE_ADAPTIVE_DETECTION = True
ADAPTIVE_PRESCAN_FRAMES = 5
ADAPTIVE_MIN_PEAKS_PER_FRAME = 180
ADAPTIVE_MAX_PEAKS_PER_FRAME = 850
ADAPTIVE_MEAN_SLOPE = 0.65
ADAPTIVE_MEAN_INTERCEPT = 130.0
ADAPTIVE_HIGH_BACKGROUND_P50 = 1200.0
ADAPTIVE_HIGH_BACKGROUND_CAP = 330

# Move selected peak coordinates to local intensity-weighted blob centroids.
ENABLE_CENTROID_REFINEMENT = True
CENTROID_ALPHA = 0.25
CENTROID_BACKGROUND_PERCENTILE = 20.0
CENTROID_CROP_RADIUS_ZYX = (5, 10, 10)
"""


DETECT_FRAME_OLD = """def detect_frame(volume: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=GAUSSIAN_SIGMA_ZYX)
    positive = smoothed[smoothed > 0]
    if len(positive) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    threshold = float(np.percentile(positive, LOW_THRESHOLD_QUANTILE))
    peaks = peak_local_max(
        smoothed,
        min_distance=MIN_PEAK_DISTANCE,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=TARGET_PEAKS_PER_FRAME,
    )
    if len(peaks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    order = np.argsort(-intensities)
    peaks = peaks[order]
    intensities = intensities[order]
    return np.column_stack([peaks, intensities]).astype(np.float32)
"""


DETECT_FRAME_NEW = """def estimate_detection_settings(image: SimpleZarr3Array) -> tuple[float, int]:
    if not ENABLE_ADAPTIVE_DETECTION:
        return LOW_THRESHOLD_QUANTILE, TARGET_PEAKS_PER_FRAME

    frame_count = int(image.shape[0])
    prescan_count = max(1, min(ADAPTIVE_PRESCAN_FRAMES, frame_count))
    frame_indices = np.linspace(0, frame_count - 1, prescan_count, dtype=int)
    stats = []
    for t in frame_indices:
        volume = np.asarray(image[int(t), :, :, :])
        positive = volume[volume > 0]
        if len(positive) == 0:
            continue
        stats.append((
            float(np.percentile(positive, 50)),
            float(np.percentile(positive, 99)),
            float(np.mean(positive)),
        ))

    if not stats:
        return LOW_THRESHOLD_QUANTILE, TARGET_PEAKS_PER_FRAME

    med_p50, med_p99, med_mean = np.median(np.asarray(stats, dtype=np.float32), axis=0)
    adaptive_cap = int(round(ADAPTIVE_MEAN_INTERCEPT + ADAPTIVE_MEAN_SLOPE * float(med_mean)))
    adaptive_cap = int(np.clip(adaptive_cap, ADAPTIVE_MIN_PEAKS_PER_FRAME, ADAPTIVE_MAX_PEAKS_PER_FRAME))
    if med_p50 >= ADAPTIVE_HIGH_BACKGROUND_P50 and (med_p99 - med_p50) <= 1400:
        adaptive_cap = min(adaptive_cap, ADAPTIVE_HIGH_BACKGROUND_CAP)

    print(
        'adaptive detection: '
        f'p50={med_p50:.1f}, p99={med_p99:.1f}, mean={med_mean:.1f}, '
        f'threshold_q={LOW_THRESHOLD_QUANTILE:.1f}, max_peaks={adaptive_cap}'
    )
    return LOW_THRESHOLD_QUANTILE, adaptive_cap


def local_blob_voxel_count(smoothed: np.ndarray, peak_zyx: np.ndarray) -> int:
    center = peak_zyx.astype(np.int64)
    radius = np.asarray(BLOB_CROP_RADIUS_ZYX, dtype=np.int64)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, np.asarray(smoothed.shape, dtype=np.int64))
    crop = smoothed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local_peak = center - lo
    peak_value = float(crop[tuple(local_peak)])
    background = float(np.percentile(crop, BLOB_BACKGROUND_PERCENTILE))
    threshold = background + BLOB_ALPHA * (peak_value - background)
    mask = crop >= threshold
    labels, label_count = connected_components(mask)
    if label_count == 0:
        return 0
    component_id = int(labels[tuple(local_peak)])
    if component_id == 0:
        return 0
    return int(np.count_nonzero(labels == component_id))


def refine_peak_centroids(smoothed: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    refined = []
    grid_cache = {}
    for peak in peaks:
        center = peak.astype(np.int64)
        radius = np.asarray(CENTROID_CROP_RADIUS_ZYX, dtype=np.int64)
        lo = np.maximum(center - radius, 0)
        hi = np.minimum(center + radius + 1, np.asarray(smoothed.shape, dtype=np.int64))
        crop = smoothed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        local_peak = center - lo
        peak_value = float(crop[tuple(local_peak)])
        background = float(np.percentile(crop, CENTROID_BACKGROUND_PERCENTILE))
        threshold = background + CENTROID_ALPHA * (peak_value - background)
        mask = crop >= threshold
        labels, label_count = connected_components(mask)
        if label_count == 0:
            refined.append(peak.astype(np.float32))
            continue
        component_id = int(labels[tuple(local_peak)])
        if component_id == 0:
            refined.append(peak.astype(np.float32))
            continue
        component = labels == component_id
        weights = np.maximum(crop.astype(np.float32) - background, 0.0) * component
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            refined.append(peak.astype(np.float32))
            continue
        shape = tuple(int(v) for v in crop.shape)
        coords = grid_cache.get(shape)
        if coords is None:
            coords = np.indices(shape, dtype=np.float32)
            grid_cache[shape] = coords
        refined.append(np.asarray([
            float((weights * coords[0]).sum() / weight_sum + lo[0]),
            float((weights * coords[1]).sum() / weight_sum + lo[1]),
            float((weights * coords[2]).sum() / weight_sum + lo[2]),
        ], dtype=np.float32))
    if not refined:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(refined, dtype=np.float32)


def rank_peaks_by_soft_blob_size(
    smoothed: np.ndarray,
    peaks: np.ndarray,
    intensities: np.ndarray,
    target_peaks_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_intensities = np.log1p(np.maximum(intensities.astype(np.float64), 0.0))
    intensity_scale = float(np.std(log_intensities)) or 1.0
    candidates = []
    for peak, intensity in zip(peaks, intensities):
        voxel_count = local_blob_voxel_count(smoothed, peak.astype(np.int64))
        size_z = (float(voxel_count) - BLOB_TARGET_VOXELS) / max(float(BLOB_SIZE_SIGMA_VOXELS), 1.0)
        size_penalty = BLOB_SIZE_PENALTY_WEIGHT * size_z * size_z
        score = np.log1p(max(float(intensity), 0.0)) / intensity_scale - size_penalty
        candidates.append((float(score), peak, float(intensity)))
    if not candidates:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:target_peaks_per_frame]
    return (
        np.asarray([item[1] for item in selected]),
        np.asarray([item[2] for item in selected], dtype=np.float32),
    )


def detect_frame(
    volume: np.ndarray,
    threshold_quantile: float,
    target_peaks_per_frame: int,
) -> np.ndarray:
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=GAUSSIAN_SIGMA_ZYX)
    positive = smoothed[smoothed > 0]
    if len(positive) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    threshold = float(np.percentile(positive, threshold_quantile))
    num_peaks = target_peaks_per_frame
    if ENABLE_BLOB_SIZE_RANKING:
        num_peaks = max(target_peaks_per_frame, int(round(target_peaks_per_frame * BLOB_RANKING_OVERSAMPLE_FACTOR)))
    peaks = peak_local_max(
        smoothed,
        min_distance=MIN_PEAK_DISTANCE,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=num_peaks,
    )
    if len(peaks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    order = np.argsort(-intensities)
    peaks = peaks[order]
    intensities = intensities[order]
    if ENABLE_BLOB_SIZE_RANKING:
        peaks, intensities = rank_peaks_by_soft_blob_size(smoothed, peaks, intensities, target_peaks_per_frame)
    else:
        peaks = peaks[:target_peaks_per_frame]
        intensities = intensities[:target_peaks_per_frame]
    if ENABLE_CENTROID_REFINEMENT:
        peaks = refine_peak_centroids(smoothed, peaks)
    return np.column_stack([peaks, intensities]).astype(np.float32)
"""


BUILD_NODES_OLD = """def build_nodes(zarr_path: Path, array_path: str = '0') -> list[dict]:
    image = open_zarr_array(zarr_path, array_path)
    nodes = []
    next_id = 1
    for t in range(int(image.shape[0])):
        peaks = detect_frame(np.asarray(image[t, :, :, :]))
        if t % 10 == 0 or t == int(image.shape[0]) - 1:
            print(f'{zarr_path.stem} t={t:03d}: {len(peaks)} peaks')
        for z, y, x, score in peaks:
            nodes.append({
                'node_id': next_id,
                't': int(t),
                'z': int(round(float(z))),
                'y': int(round(float(y))),
                'x': int(round(float(x))),
                'score': float(score),
            })
            next_id += 1
    return nodes
"""


BUILD_NODES_NEW = """def build_nodes(zarr_path: Path, array_path: str = '0') -> list[dict]:
    image = open_zarr_array(zarr_path, array_path)
    threshold_quantile, target_peaks_per_frame = estimate_detection_settings(image)
    nodes = []
    next_id = 1
    for t in range(int(image.shape[0])):
        peaks = detect_frame(np.asarray(image[t, :, :, :]), threshold_quantile, target_peaks_per_frame)
        if t % 10 == 0 or t == int(image.shape[0]) - 1:
            print(f'{zarr_path.stem} t={t:03d}: {len(peaks)} peaks')
        for z, y, x, score in peaks:
            nodes.append({
                'node_id': next_id,
                't': int(t),
                'z': int(round(float(z))),
                'y': int(round(float(y))),
                'x': int(round(float(x))),
                'score': float(score),
            })
            next_id += 1
    return nodes
"""


def main() -> None:
    nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    nb["cells"][0]["source"] = [
        "# Biohub Cell Tracking - Adaptive Centroid Soft Ellipsoid Classical Submission\n",
        "\n",
        "Classical peak detection with adaptive per-sample caps, centroid refinement, global-flow linking, and soft blob-size/ellipsoid ranking.\n",
    ]

    imports = "".join(nb["cells"][1]["source"])
    imports = imports.replace("from scipy.ndimage import gaussian_filter", "from scipy.ndimage import gaussian_filter, label as connected_components")
    nb["cells"][1]["source"] = imports.splitlines(keepends=True)

    config = "".join(nb["cells"][2]["source"])
    if "ENABLE_GLOBAL_FLOW" not in config:
        config = config.rstrip() + CONFIG_APPEND
    if "ENABLE_BLOB_SIZE_RANKING" not in config:
        config = config.rstrip() + CONFIG_APPEND
    nb["cells"][2]["source"] = config.splitlines(keepends=True)

    code = "".join(nb["cells"][5]["source"])
    if DETECT_FRAME_OLD not in code:
        raise RuntimeError("Could not find detect_frame block to replace")
    code = code.replace(DETECT_FRAME_OLD, DETECT_FRAME_NEW)
    if BUILD_NODES_OLD not in code:
        raise RuntimeError("Could not find build_nodes block to replace")
    code = code.replace(BUILD_NODES_OLD, BUILD_NODES_NEW)
    nb["cells"][5]["source"] = code.splitlines(keepends=True)

    OUTPUT_NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    joined_code = "\n\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"] if cell.get("cell_type") == "code"
    )
    tmp_path = Path(tempfile.gettempdir()) / (OUTPUT_NOTEBOOK.name + ".py")
    tmp_path.write_text(joined_code, encoding="utf-8")
    py_compile.compile(str(tmp_path), doraise=True)
    print(f"wrote and compiled {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
