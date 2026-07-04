from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label as connected_components
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

from simple_zarr import open_array, open_group


SCALE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical 3D peak detector + adjacent-frame linker for one Biohub sample."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to one .zarr sample.")
    parser.add_argument("--geff", type=Path, default=None, help="Optional matching .geff.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/classical_one"))
    parser.add_argument("--array-path", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="Print and save stage timing information.")
    parser.add_argument("--threshold-quantile", type=float, default=99.92)
    parser.add_argument("--min-peak-distance", type=int, default=4)
    parser.add_argument("--max-peaks-per-frame", type=int, default=1500)
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")
    parser.add_argument(
        "--contrast-mode",
        choices=["raw", "subtract", "ratio"],
        default="raw",
        help="Optional local contrast normalization before peak finding.",
    )
    parser.add_argument("--background-sigma-zyx", default="3.0,8.0,8.0")
    parser.add_argument("--contrast-ratio-epsilon", type=float, default=50.0)
    parser.add_argument(
        "--enable-adaptive-detection",
        action="store_true",
        help="Prescan a few frames and adapt the per-frame peak cap to sample brightness.",
    )
    parser.add_argument("--adaptive-prescan-frames", type=int, default=5)
    parser.add_argument("--adaptive-min-peaks-per-frame", type=int, default=180)
    parser.add_argument("--adaptive-max-peaks-per-frame", type=int, default=850)
    parser.add_argument("--adaptive-mean-slope", type=float, default=0.65)
    parser.add_argument("--adaptive-mean-intercept", type=float, default=130.0)
    parser.add_argument("--adaptive-high-background-p50", type=float, default=1200.0)
    parser.add_argument("--adaptive-high-background-cap", type=int, default=330)
    parser.add_argument(
        "--enable-blob-size-filter",
        action="store_true",
        help="Reject peak candidates whose local connected component has implausible cell size.",
    )
    parser.add_argument("--blob-size-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--blob-min-voxels", type=int, default=450)
    parser.add_argument("--blob-max-voxels", type=int, default=1400)
    parser.add_argument("--blob-target-voxels", type=float, default=928.0)
    parser.add_argument("--blob-size-sigma-voxels", type=float, default=350.0)
    parser.add_argument("--blob-size-penalty-weight", type=float, default=1.0)
    parser.add_argument("--blob-filter-oversample-factor", type=float, default=2.0)
    parser.add_argument("--blob-alpha", type=float, default=0.35)
    parser.add_argument("--blob-background-percentile", type=float, default=20.0)
    parser.add_argument("--blob-crop-radius-zyx", default="4,8,8")
    parser.add_argument(
        "--enable-centroid-refinement",
        action="store_true",
        help="Move selected peaks to an intensity-weighted local blob centroid before output/linking.",
    )
    parser.add_argument("--centroid-alpha", type=float, default=0.25)
    parser.add_argument("--centroid-background-percentile", type=float, default=20.0)
    parser.add_argument("--centroid-crop-radius-zyx", default="5,10,10")
    parser.add_argument(
        "--proposal-scorer-path",
        type=Path,
        default=None,
        help="Optional TinyPatchScorer .pt checkpoint used to rescore/rank peak proposals.",
    )
    parser.add_argument("--proposal-scorer-threshold", type=float, default=0.5)
    parser.add_argument("--proposal-scorer-oversample-factor", type=float, default=3.0)
    parser.add_argument("--proposal-scorer-batch-size", type=int, default=256)
    parser.add_argument("--proposal-scorer-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--proposal-scorer-min-keep-fraction",
        type=float,
        default=0.5,
        help=(
            "If thresholding is too strict, still keep at least this fraction of max_peaks "
            "from the top CNN-ranked candidates."
        ),
    )
    parser.add_argument("--link-max-distance-um", type=float, default=7.0)
    parser.add_argument(
        "--enable-global-flow",
        action="store_true",
        help="Estimate a median frame-to-frame displacement and use it to predict link targets.",
    )
    parser.add_argument(
        "--flow-confident-distance-um",
        type=float,
        default=4.0,
        help="Initial links at or below this distance contribute to the global flow estimate.",
    )
    parser.add_argument("--enable-divisions", action="store_true")
    parser.add_argument("--division-max-parent-distance-um", type=float, default=10.0)
    parser.add_argument("--division-min-daughter-separation-um", type=float, default=2.0)
    parser.add_argument("--division-max-daughter-separation-um", type=float, default=15.0)
    parser.add_argument("--division-max-midpoint-distance-um", type=float, default=4.0)
    parser.add_argument("--division-max-nearby-candidates", type=int, default=2)
    parser.add_argument("--division-persistence-frames", type=int, default=3)
    parser.add_argument(
        "--division-allow-parent-rewrite",
        action="store_true",
        help="Allow division hypotheses to replace an existing one-child parent continuation.",
    )
    return parser.parse_args()


def parse_sigma(text: str) -> tuple[float, float, float]:
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--gaussian-sigma-zyx must have 3 comma-separated values")
    return tuple(parts)


def parse_int_triplet(text: str, option_name: str) -> tuple[int, int, int]:
    parts = [int(round(float(p.strip()))) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{option_name} must have 3 comma-separated values")
    return tuple(parts)


def load_geff(geff_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = open_group(geff_path, mode="r")
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


def load_estimated_number_of_nodes(geff_path: Path) -> int | None:
    root = open_group(geff_path, mode="r")
    geff_attrs = root.attrs.get("geff", {})
    extra = geff_attrs.get("extra", {})
    estimated = extra.get("estimated_number_of_nodes")
    return int(estimated) if estimated is not None else None


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32, copy=False)
    low = float(np.percentile(patch, 10))
    high = float(np.percentile(patch, 99))
    scale = max(high - low, 1.0)
    return np.clip((patch - low) / scale, 0.0, 1.0).astype(np.float32)


def crop_patch(volume: np.ndarray, center_zyx: np.ndarray, radius_zyx: tuple[int, int, int]) -> np.ndarray | None:
    center = np.asarray(np.rint(center_zyx), dtype=np.int64)
    radius = np.asarray(radius_zyx, dtype=np.int64)
    lo = center - radius
    hi = center + radius + 1
    shape = np.asarray(volume.shape, dtype=np.int64)
    if np.any(lo < 0) or np.any(hi > shape):
        return None
    return volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


@dataclass
class ProposalScorer:
    torch: object
    model: object
    device: object
    patch_radius_zyx: tuple[int, int, int]


def load_proposal_scorer(checkpoint_path: Path, device_name: str) -> ProposalScorer:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for --proposal-scorer-path. "
            "Install torch or run without the proposal scorer."
        ) from exc

    class TinyPatchScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((2, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.net(x)).squeeze(1)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyPatchScorer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    patch_radius_zyx = tuple(int(v) for v in checkpoint.get("patch_radius_zyx", (5, 10, 10)))
    print(
        f"loaded proposal scorer {checkpoint_path} on {device}; "
        f"patch_radius_zyx={patch_radius_zyx}, best_epoch={checkpoint.get('best_epoch')}"
    )
    return ProposalScorer(torch=torch, model=model, device=device, patch_radius_zyx=patch_radius_zyx)


def rescore_proposals_with_cnn(
    volume: np.ndarray,
    peaks: np.ndarray,
    intensities: np.ndarray,
    scorer: ProposalScorer,
    max_peaks: int,
    threshold: float,
    batch_size: int,
    min_keep_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(peaks) == 0:
        return peaks, intensities.astype(np.float32)

    valid_indices: list[int] = []
    patches: list[np.ndarray] = []
    for idx, peak in enumerate(peaks):
        patch = crop_patch(volume, peak, scorer.patch_radius_zyx)
        if patch is None:
            continue
        valid_indices.append(idx)
        patches.append(normalize_patch(patch))

    if not patches:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    torch = scorer.torch
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = np.stack(patches[start : start + batch_size], axis=0)
            x = torch.from_numpy(batch[:, None, :, :, :]).to(scorer.device)
            logits = scorer.model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
            scores.append(probs)

    cnn_scores = np.concatenate(scores)
    valid_peaks = peaks[np.asarray(valid_indices, dtype=np.int64)].astype(np.float32)
    valid_intensities = intensities[np.asarray(valid_indices, dtype=np.int64)].astype(np.float32)
    order = np.lexsort((-valid_intensities, -cnn_scores))

    keep_mask = cnn_scores[order] >= threshold
    min_keep = int(round(max_peaks * max(0.0, min(min_keep_fraction, 1.0))))
    keep_count = min(max_peaks, max(int(np.count_nonzero(keep_mask)), min_keep))
    selected = order[:keep_count]
    return valid_peaks[selected], cnn_scores[selected].astype(np.float32)


def local_blob_voxel_count(
    smoothed: np.ndarray,
    peak_zyx: np.ndarray,
    crop_radius_zyx: tuple[int, int, int],
    alpha: float,
    background_percentile: float,
) -> int:
    center = peak_zyx.astype(np.int64)
    radius = np.asarray(crop_radius_zyx, dtype=np.int64)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, np.asarray(smoothed.shape, dtype=np.int64))
    crop = smoothed[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    local_peak = center - lo
    peak_value = float(crop[tuple(local_peak)])
    background = float(np.percentile(crop, background_percentile))
    threshold = background + alpha * (peak_value - background)
    mask = crop >= threshold
    labels, label_count = connected_components(mask)
    if label_count == 0:
        return 0
    component_id = int(labels[tuple(local_peak)])
    if component_id == 0:
        return 0
    return int(np.count_nonzero(labels == component_id))


def filter_peaks_by_blob_size(
    smoothed: np.ndarray,
    peaks: np.ndarray,
    intensities: np.ndarray,
    max_peaks: int,
    mode: str,
    min_voxels: int,
    max_voxels: int,
    target_voxels: float,
    size_sigma_voxels: float,
    penalty_weight: float,
    crop_radius_zyx: tuple[int, int, int],
    alpha: float,
    background_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    log_intensities = np.log1p(np.maximum(intensities.astype(np.float64), 0.0))
    intensity_scale = float(np.std(log_intensities)) or 1.0
    for peak, intensity in zip(peaks, intensities):
        voxel_count = local_blob_voxel_count(
            smoothed,
            peak.astype(np.int64),
            crop_radius_zyx=crop_radius_zyx,
            alpha=alpha,
            background_percentile=background_percentile,
        )
        if mode == "hard":
            if min_voxels <= voxel_count <= max_voxels:
                candidates.append((float(intensity), peak, float(intensity)))
                if len(candidates) >= max_peaks:
                    break
        else:
            sigma = max(float(size_sigma_voxels), 1.0)
            size_z = (float(voxel_count) - float(target_voxels)) / sigma
            size_penalty = penalty_weight * size_z * size_z
            score = np.log1p(max(float(intensity), 0.0)) / intensity_scale - size_penalty
            candidates.append((float(score), peak, float(intensity)))

    if not candidates:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_peaks]
    return (
        np.asarray([item[1] for item in selected]),
        np.asarray([item[2] for item in selected], dtype=np.float32),
    )


def refine_peak_centroids(
    smoothed: np.ndarray,
    peaks: np.ndarray,
    crop_radius_zyx: tuple[int, int, int],
    alpha: float,
    background_percentile: float,
) -> np.ndarray:
    refined = []
    grid_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for peak in peaks:
        center = peak.astype(np.int64)
        radius = np.asarray(crop_radius_zyx, dtype=np.int64)
        lo = np.maximum(center - radius, 0)
        hi = np.minimum(center + radius + 1, np.asarray(smoothed.shape, dtype=np.int64))
        crop = smoothed[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        local_peak = center - lo
        peak_value = float(crop[tuple(local_peak)])
        background = float(np.percentile(crop, background_percentile))
        threshold = background + alpha * (peak_value - background)
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
        centroid = np.asarray(
            [
                float((weights * coords[0]).sum() / weight_sum + lo[0]),
                float((weights * coords[1]).sum() / weight_sum + lo[1]),
                float((weights * coords[2]).sum() / weight_sum + lo[2]),
            ],
            dtype=np.float32,
        )
        refined.append(centroid)
    if not refined:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(refined, dtype=np.float32)


def preprocess_for_detection(
    volume: np.ndarray,
    sigma_zyx: tuple[float, float, float],
    contrast_mode: str,
    background_sigma_zyx: tuple[float, float, float],
    ratio_epsilon: float,
) -> np.ndarray:
    small = gaussian_filter(volume.astype(np.float32), sigma=sigma_zyx)
    if contrast_mode == "raw":
        return small

    background = gaussian_filter(volume.astype(np.float32), sigma=background_sigma_zyx)
    enhanced = small - background
    if contrast_mode == "ratio":
        enhanced = enhanced / np.maximum(background, float(ratio_epsilon))
    return np.maximum(enhanced, 0.0).astype(np.float32, copy=False)


def estimate_adaptive_detection_settings(image, args: argparse.Namespace, frame_count: int) -> tuple[float, int]:
    if not args.enable_adaptive_detection or frame_count <= 0:
        return args.threshold_quantile, args.max_peaks_per_frame

    prescan_count = max(1, min(args.adaptive_prescan_frames, frame_count))
    frame_indices = np.linspace(0, frame_count - 1, prescan_count, dtype=int)
    stats = []
    for t in frame_indices:
        volume = np.asarray(image[int(t), :, :, :])
        positive = volume[volume > 0]
        if len(positive) == 0:
            continue
        stats.append(
            (
                float(np.percentile(positive, 50)),
                float(np.percentile(positive, 99)),
                float(np.mean(positive)),
            )
        )

    if not stats:
        return args.threshold_quantile, args.max_peaks_per_frame

    med_p50, med_p99, med_mean = np.median(np.asarray(stats, dtype=np.float32), axis=0)
    adaptive_cap = int(round(args.adaptive_mean_intercept + args.adaptive_mean_slope * float(med_mean)))
    adaptive_cap = int(np.clip(adaptive_cap, args.adaptive_min_peaks_per_frame, args.adaptive_max_peaks_per_frame))

    # Very bright low-contrast fields behave like a high background rather than a high cell count.
    if med_p50 >= args.adaptive_high_background_p50 and (med_p99 - med_p50) <= 1400:
        adaptive_cap = min(adaptive_cap, args.adaptive_high_background_cap)

    print(
        "adaptive detection: "
        f"p50={med_p50:.1f}, p99={med_p99:.1f}, mean={med_mean:.1f}, "
        f"threshold_q={args.threshold_quantile:.1f}, max_peaks={adaptive_cap}"
    )
    return args.threshold_quantile, adaptive_cap


def detect_frame(
    volume: np.ndarray,
    sigma_zyx: tuple[float, float, float],
    threshold_quantile: float,
    min_peak_distance: int,
    max_peaks: int,
    contrast_mode: str = "raw",
    background_sigma_zyx: tuple[float, float, float] = (3.0, 8.0, 8.0),
    contrast_ratio_epsilon: float = 50.0,
    enable_blob_size_filter: bool = False,
    blob_size_mode: str = "hard",
    blob_min_voxels: int = 450,
    blob_max_voxels: int = 1400,
    blob_target_voxels: float = 928.0,
    blob_size_sigma_voxels: float = 350.0,
    blob_size_penalty_weight: float = 1.0,
    blob_filter_oversample_factor: float = 2.0,
    blob_alpha: float = 0.35,
    blob_background_percentile: float = 20.0,
    blob_crop_radius_zyx: tuple[int, int, int] = (4, 8, 8),
    enable_centroid_refinement: bool = False,
    centroid_alpha: float = 0.25,
    centroid_background_percentile: float = 20.0,
    centroid_crop_radius_zyx: tuple[int, int, int] = (5, 10, 10),
    proposal_scorer: ProposalScorer | None = None,
    proposal_scorer_threshold: float = 0.5,
    proposal_scorer_oversample_factor: float = 3.0,
    proposal_scorer_batch_size: int = 256,
    proposal_scorer_min_keep_fraction: float = 0.5,
) -> np.ndarray:
    smoothed = preprocess_for_detection(
        volume,
        sigma_zyx=sigma_zyx,
        contrast_mode=contrast_mode,
        background_sigma_zyx=background_sigma_zyx,
        ratio_epsilon=contrast_ratio_epsilon,
    )
    positive = smoothed[smoothed > 0]
    if len(positive) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    threshold = float(np.percentile(positive, threshold_quantile))
    proposal_count = max_peaks
    if enable_blob_size_filter:
        proposal_count = max(proposal_count, int(round(max_peaks * blob_filter_oversample_factor)))
    if proposal_scorer is not None:
        proposal_count = max(proposal_count, int(round(max_peaks * proposal_scorer_oversample_factor)))

    peaks = peak_local_max(
        smoothed,
        min_distance=min_peak_distance,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=proposal_count,
    )
    if len(peaks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    intensities = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    order = np.argsort(-intensities)
    peaks = peaks[order]
    intensities = intensities[order]
    if enable_blob_size_filter:
        peaks, intensities = filter_peaks_by_blob_size(
            smoothed,
            peaks,
            intensities,
            max_peaks=proposal_count if proposal_scorer is not None else max_peaks,
            mode=blob_size_mode,
            min_voxels=blob_min_voxels,
            max_voxels=blob_max_voxels,
            target_voxels=blob_target_voxels,
            size_sigma_voxels=blob_size_sigma_voxels,
            penalty_weight=blob_size_penalty_weight,
            crop_radius_zyx=blob_crop_radius_zyx,
            alpha=blob_alpha,
            background_percentile=blob_background_percentile,
        )
    elif proposal_scorer is None:
        peaks = peaks[:max_peaks]
        intensities = intensities[:max_peaks]
    if proposal_scorer is not None:
        peaks, intensities = rescore_proposals_with_cnn(
            volume,
            peaks,
            intensities,
            scorer=proposal_scorer,
            max_peaks=max_peaks,
            threshold=proposal_scorer_threshold,
            batch_size=proposal_scorer_batch_size,
            min_keep_fraction=proposal_scorer_min_keep_fraction,
        )
    if enable_centroid_refinement:
        peaks = refine_peak_centroids(
            smoothed,
            peaks,
            crop_radius_zyx=centroid_crop_radius_zyx,
            alpha=centroid_alpha,
            background_percentile=centroid_background_percentile,
        )
    return np.column_stack([peaks, intensities]).astype(np.float32)


def build_nodes(image, args: argparse.Namespace) -> list[dict]:
    sigma_zyx = parse_sigma(args.gaussian_sigma_zyx)
    background_sigma_zyx = parse_sigma(args.background_sigma_zyx)
    blob_crop_radius_zyx = parse_int_triplet(args.blob_crop_radius_zyx, "--blob-crop-radius-zyx")
    centroid_crop_radius_zyx = parse_int_triplet(args.centroid_crop_radius_zyx, "--centroid-crop-radius-zyx")
    total_frames = int(image.shape[0])
    frame_count = total_frames if args.max_frames is None else min(args.max_frames, total_frames)
    threshold_quantile, max_peaks_per_frame = estimate_adaptive_detection_settings(image, args, frame_count)
    proposal_scorer = None
    if args.proposal_scorer_path is not None:
        proposal_scorer = load_proposal_scorer(args.proposal_scorer_path, args.proposal_scorer_device)

    nodes: list[dict] = []
    next_id = 1
    for t in range(frame_count):
        volume = np.asarray(image[t, :, :, :])
        peaks = detect_frame(
            volume,
            sigma_zyx=sigma_zyx,
            contrast_mode=args.contrast_mode,
            background_sigma_zyx=background_sigma_zyx,
            contrast_ratio_epsilon=args.contrast_ratio_epsilon,
            threshold_quantile=threshold_quantile,
            min_peak_distance=args.min_peak_distance,
            max_peaks=max_peaks_per_frame,
            enable_blob_size_filter=args.enable_blob_size_filter,
            blob_size_mode=args.blob_size_mode,
            blob_min_voxels=args.blob_min_voxels,
            blob_max_voxels=args.blob_max_voxels,
            blob_target_voxels=args.blob_target_voxels,
            blob_size_sigma_voxels=args.blob_size_sigma_voxels,
            blob_size_penalty_weight=args.blob_size_penalty_weight,
            blob_filter_oversample_factor=args.blob_filter_oversample_factor,
            blob_alpha=args.blob_alpha,
            blob_background_percentile=args.blob_background_percentile,
            blob_crop_radius_zyx=blob_crop_radius_zyx,
            enable_centroid_refinement=args.enable_centroid_refinement,
            centroid_alpha=args.centroid_alpha,
            centroid_background_percentile=args.centroid_background_percentile,
            centroid_crop_radius_zyx=centroid_crop_radius_zyx,
            proposal_scorer=proposal_scorer,
            proposal_scorer_threshold=args.proposal_scorer_threshold,
            proposal_scorer_oversample_factor=args.proposal_scorer_oversample_factor,
            proposal_scorer_batch_size=args.proposal_scorer_batch_size,
            proposal_scorer_min_keep_fraction=args.proposal_scorer_min_keep_fraction,
        )
        print(f"t={t:03d}: {len(peaks)} peaks")
        for z, y, x, score in peaks:
            nodes.append(
                {
                    "node_id": next_id,
                    "t": int(t),
                    "z": int(round(float(z))),
                    "y": int(round(float(y))),
                    "x": int(round(float(x))),
                    "score": float(score),
                }
            )
            next_id += 1
    return nodes


def solve_frame_links(
    prev_xyz: np.ndarray,
    curr_xyz: np.ndarray,
    max_distance_um: float,
    flow_um: np.ndarray | None = None,
) -> list[tuple[int, int, float, float]]:
    if flow_um is None:
        flow_um = np.zeros(3, dtype=np.float32)
    predicted_xyz = prev_xyz + flow_um
    residual = np.linalg.norm(predicted_xyz[:, None, :] - curr_xyz[None, :, :], axis=2)
    cost = residual.copy()
    cost[cost > max_distance_um] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)

    links: list[tuple[int, int, float, float]] = []
    actual_dist = np.linalg.norm(prev_xyz[:, None, :] - curr_xyz[None, :, :], axis=2)
    for r, c in zip(row_ind, col_ind):
        residual_distance = float(residual[r, c])
        if residual_distance <= max_distance_um:
            links.append((int(r), int(c), float(actual_dist[r, c]), residual_distance))
    return links


def estimate_global_flow_um(
    prev_xyz: np.ndarray,
    curr_xyz: np.ndarray,
    max_distance_um: float,
    confident_distance_um: float,
) -> np.ndarray:
    initial_links = solve_frame_links(prev_xyz, curr_xyz, max_distance_um)
    displacements = []
    for r, c, actual_distance, _ in initial_links:
        if actual_distance <= confident_distance_um:
            displacements.append(curr_xyz[c] - prev_xyz[r])
    if not displacements:
        return np.zeros(3, dtype=np.float32)
    return np.median(np.asarray(displacements, dtype=np.float32), axis=0).astype(np.float32)


def link_nodes(
    nodes: list[dict],
    max_distance_um: float,
    enable_global_flow: bool = False,
    flow_confident_distance_um: float = 4.0,
) -> list[tuple[int, int, float]]:
    by_t: dict[int, list[dict]] = defaultdict(list)
    for node in nodes:
        by_t[node["t"]].append(node)

    edges: list[tuple[int, int, float]] = []
    times = sorted(by_t)
    for t0, t1 in zip(times[:-1], times[1:]):
        if t1 != t0 + 1:
            continue
        prev = by_t[t0]
        curr = by_t[t1]
        if not prev or not curr:
            continue

        prev_xyz = np.asarray([[n["z"], n["y"], n["x"]] for n in prev], dtype=np.float32) * SCALE_ZYX
        curr_xyz = np.asarray([[n["z"], n["y"], n["x"]] for n in curr], dtype=np.float32) * SCALE_ZYX
        flow_um = np.zeros(3, dtype=np.float32)
        if enable_global_flow:
            flow_um = estimate_global_flow_um(
                prev_xyz,
                curr_xyz,
                max_distance_um=max_distance_um,
                confident_distance_um=flow_confident_distance_um,
            )

        links = solve_frame_links(prev_xyz, curr_xyz, max_distance_um, flow_um=flow_um)
        for r, c, actual_distance, _ in links:
            edges.append((prev[r]["node_id"], curr[c]["node_id"], actual_distance))
        if enable_global_flow:
            print(
                "link "
                f"t={t0:03d}->{t1:03d}: {len(links)} edges, "
                f"flow_um=({flow_um[0]:.3f},{flow_um[1]:.3f},{flow_um[2]:.3f})"
            )
        else:
            print(f"link t={t0:03d}->{t1:03d}: {len(links)} edges")
    return edges


def build_edge_maps(
    edges: list[tuple[int, int, float]],
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    outgoing: dict[int, list[tuple[int, float]]] = defaultdict(list)
    incoming: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for source, target, distance in edges:
        outgoing[source].append((target, distance))
        incoming[target].append((source, distance))
    return outgoing, incoming


def scaled_distance(a: dict, b: dict) -> float:
    a_xyz = np.asarray([a["z"], a["y"], a["x"]], dtype=np.float32) * SCALE_ZYX
    b_xyz = np.asarray([b["z"], b["y"], b["x"]], dtype=np.float32) * SCALE_ZYX
    return float(np.linalg.norm(a_xyz - b_xyz))


def forward_track_length(node_id: int, outgoing: dict[int, list[tuple[int, float]]], limit: int) -> int:
    length = 1
    current = node_id
    while length < limit:
        next_edges = outgoing.get(current, [])
        if len(next_edges) != 1:
            break
        current = next_edges[0][0]
        length += 1
    return length


def add_division_edges(
    nodes: list[dict],
    edges: list[tuple[int, int, float]],
    max_parent_distance_um: float,
    min_daughter_separation_um: float,
    max_daughter_separation_um: float,
    max_midpoint_distance_um: float,
    max_nearby_candidates: int,
    persistence_frames: int,
    allow_parent_rewrite: bool,
) -> tuple[list[tuple[int, int, float]], list[dict]]:
    node_by_id = {node["node_id"]: node for node in nodes}
    by_t: dict[int, list[dict]] = defaultdict(list)
    for node in nodes:
        by_t[node["t"]].append(node)

    outgoing, incoming = build_edge_maps(edges)
    edge_by_pair = {(source, target): distance for source, target, distance in edges}
    accepted: list[dict] = []
    used_parents: set[int] = set()
    used_daughters: set[int] = set()

    for t in sorted(by_t):
        parents = by_t.get(t, [])
        daughters = by_t.get(t + 1, [])
        if not parents or len(daughters) < 2:
            continue

        for parent in parents:
            parent_id = parent["node_id"]
            if parent_id in used_parents:
                continue

            current_targets = {target for target, _ in outgoing.get(parent_id, [])}
            if len(current_targets) > 1:
                continue
            if current_targets and not allow_parent_rewrite:
                continue

            nearby = []
            for daughter in daughters:
                daughter_id = daughter["node_id"]
                if daughter_id in used_daughters:
                    continue
                distance = scaled_distance(parent, daughter)
                if distance <= max_parent_distance_um:
                    nearby.append((daughter, distance))

            if len(nearby) < 2:
                continue
            if max_nearby_candidates > 0 and len(nearby) > max_nearby_candidates:
                continue

            hypotheses = []
            for i in range(len(nearby)):
                for j in range(i + 1, len(nearby)):
                    daughter_a, dist_a = nearby[i]
                    daughter_b, dist_b = nearby[j]
                    daughter_a_id = daughter_a["node_id"]
                    daughter_b_id = daughter_b["node_id"]
                    daughter_set = {daughter_a_id, daughter_b_id}

                    if current_targets and not current_targets.issubset(daughter_set):
                        continue

                    incoming_a = {source for source, _ in incoming.get(daughter_a_id, [])}
                    incoming_b = {source for source, _ in incoming.get(daughter_b_id, [])}
                    if incoming_a - {parent_id} or incoming_b - {parent_id}:
                        continue
                    if not allow_parent_rewrite and (incoming_a or incoming_b):
                        continue

                    daughter_sep = scaled_distance(daughter_a, daughter_b)
                    if not (min_daughter_separation_um <= daughter_sep <= max_daughter_separation_um):
                        continue

                    parent_xyz = np.asarray([parent["z"], parent["y"], parent["x"]], dtype=np.float32) * SCALE_ZYX
                    daughter_a_xyz = (
                        np.asarray([daughter_a["z"], daughter_a["y"], daughter_a["x"]], dtype=np.float32)
                        * SCALE_ZYX
                    )
                    daughter_b_xyz = (
                        np.asarray([daughter_b["z"], daughter_b["y"], daughter_b["x"]], dtype=np.float32)
                        * SCALE_ZYX
                    )
                    midpoint_distance = float(
                        np.linalg.norm(parent_xyz - ((daughter_a_xyz + daughter_b_xyz) * 0.5))
                    )
                    if midpoint_distance > max_midpoint_distance_um:
                        continue

                    persist_a = forward_track_length(daughter_a_id, outgoing, persistence_frames)
                    persist_b = forward_track_length(daughter_b_id, outgoing, persistence_frames)
                    if persist_a < persistence_frames or persist_b < persistence_frames:
                        continue

                    score = dist_a + dist_b + abs(daughter_sep - min(daughter_sep, max_parent_distance_um))
                    hypotheses.append(
                        {
                            "score": score,
                            "parent_id": parent_id,
                            "daughter_a_id": daughter_a_id,
                            "daughter_b_id": daughter_b_id,
                            "distance_a_um": dist_a,
                            "distance_b_um": dist_b,
                            "daughter_separation_um": daughter_sep,
                            "midpoint_distance_um": midpoint_distance,
                            "persistence_a": persist_a,
                            "persistence_b": persist_b,
                            "t": parent["t"],
                        }
                    )

            if not hypotheses:
                continue

            best = min(hypotheses, key=lambda item: item["score"])
            daughter_ids = {best["daughter_a_id"], best["daughter_b_id"]}

            for target in list(current_targets):
                edge_by_pair.pop((parent_id, target), None)
            edge_by_pair[(parent_id, best["daughter_a_id"])] = best["distance_a_um"]
            edge_by_pair[(parent_id, best["daughter_b_id"])] = best["distance_b_um"]

            outgoing, incoming = build_edge_maps(
                [(source, target, distance) for (source, target), distance in edge_by_pair.items()]
            )
            used_parents.add(parent_id)
            used_daughters.update(daughter_ids)
            accepted.append(best)

    new_edges = [(source, target, distance) for (source, target), distance in edge_by_pair.items()]
    new_edges.sort(key=lambda edge: (node_by_id[edge[0]]["t"], edge[0], edge[1]))
    return new_edges, accepted


def assign_track_ids(nodes: list[dict], edges: list[tuple[int, int, float]]) -> dict[int, int]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, target, _ in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)

    node_ids = [n["node_id"] for n in nodes]
    track_ids: dict[int, int] = {}
    next_track = 1
    for node_id in node_ids:
        if node_id in track_ids:
            continue
        queue = deque([node_id])
        track_ids[node_id] = next_track
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt not in track_ids:
                    track_ids[nxt] = next_track
                    queue.append(nxt)
        next_track += 1
    return track_ids


def match_by_time(pred_nodes: list[dict], gt_nodes: np.ndarray, max_distance_um: float):
    pred_by_t: dict[int, list[dict]] = defaultdict(list)
    gt_by_t: dict[int, np.ndarray] = defaultdict(list)
    for node in pred_nodes:
        pred_by_t[node["t"]].append(node)
    for row in gt_nodes:
        gt_by_t[int(row[1])].append(row)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    distances: list[float] = []
    for t in sorted(set(pred_by_t) | set(gt_by_t)):
        preds = pred_by_t.get(t, [])
        gts = np.asarray(gt_by_t.get(t, []), dtype=np.int64)
        if len(preds) == 0 or len(gts) == 0:
            continue
        pred_xyz = np.asarray([[n["z"], n["y"], n["x"]] for n in preds], dtype=np.float32) * SCALE_ZYX
        gt_xyz = gts[:, 2:5].astype(np.float32) * SCALE_ZYX
        dist = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
        cost = dist.copy()
        cost[cost > max_distance_um] = 1e6
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            d = float(dist[r, c])
            if d <= max_distance_um:
                pred_to_gt[preds[r]["node_id"]] = int(gts[c, 0])
                gt_to_pred[int(gts[c, 0])] = preds[r]["node_id"]
                distances.append(d)
    return pred_to_gt, gt_to_pred, distances


def compute_sparse_metrics(
    pred_nodes: list[dict],
    pred_edges: list[tuple[int, int, float]],
    gt_nodes: np.ndarray,
    gt_edges: np.ndarray,
    max_distance_um: float,
    estimated_number_of_nodes: int | None = None,
) -> dict:
    pred_to_gt, gt_to_pred, distances = match_by_time(pred_nodes, gt_nodes, max_distance_um)
    gt_edge_set = {(int(s), int(t)) for s, t in gt_edges}
    pred_edge_pairs = [(int(s), int(t)) for s, t, _ in pred_edges]
    pred_outgoing: dict[int, list[int]] = defaultdict(list)
    gt_outgoing: dict[int, list[int]] = defaultdict(list)
    for source, target in pred_edge_pairs:
        pred_outgoing[source].append(target)
    for source, target in gt_edge_set:
        gt_outgoing[source].append(target)

    edge_tp = 0
    for source, target in pred_edge_pairs:
        gt_source = pred_to_gt.get(source)
        gt_target = pred_to_gt.get(target)
        if gt_source is not None and gt_target is not None and (gt_source, gt_target) in gt_edge_set:
            edge_tp += 1

    matched_pred_edges = sum(
        1 for source, target in pred_edge_pairs if source in pred_to_gt and target in pred_to_gt
    )
    pred_divisions = {source: targets for source, targets in pred_outgoing.items() if len(targets) >= 2}
    gt_divisions = {source: targets for source, targets in gt_outgoing.items() if len(targets) >= 2}
    division_tp = 0
    for pred_source, pred_targets in pred_divisions.items():
        gt_source = pred_to_gt.get(pred_source)
        if gt_source not in gt_divisions:
            continue
        matched_targets = {pred_to_gt[target] for target in pred_targets if target in pred_to_gt}
        if len(set(gt_divisions[gt_source]) & matched_targets) >= 2:
            division_tp += 1

    metrics = {
        "note": "Labels are sparse. Detection precision and edge precision are only against the labeled subset.",
        "max_match_distance_um": max_distance_um,
        "pred_nodes": len(pred_nodes),
        "estimated_number_of_nodes": estimated_number_of_nodes,
        "pred_to_estimated_node_ratio": (
            len(pred_nodes) / estimated_number_of_nodes if estimated_number_of_nodes else None
        ),
        "estimated_minus_pred_nodes": (
            estimated_number_of_nodes - len(pred_nodes) if estimated_number_of_nodes else None
        ),
        "gt_sparse_nodes": int(len(gt_nodes)),
        "matched_gt_nodes": len(gt_to_pred),
        "matched_pred_nodes": len(pred_to_gt),
        "sparse_node_recall": len(gt_to_pred) / max(1, len(gt_nodes)),
        "labeled_subset_node_fraction": len(pred_to_gt) / max(1, len(pred_nodes)),
        "mean_matched_distance_um": float(np.mean(distances)) if distances else None,
        "median_matched_distance_um": float(np.median(distances)) if distances else None,
        "pred_edges": len(pred_edges),
        "gt_sparse_edges": int(len(gt_edge_set)),
        "matched_pred_edges_with_labeled_endpoints": matched_pred_edges,
        "sparse_edge_tp": edge_tp,
        "sparse_edge_recall": edge_tp / max(1, len(gt_edge_set)),
        "labeled_subset_edge_precision": edge_tp / max(1, matched_pred_edges),
        "pred_division_nodes": len(pred_divisions),
        "gt_sparse_division_nodes": len(gt_divisions),
        "sparse_division_tp": division_tp,
        "sparse_division_recall": division_tp / max(1, len(gt_divisions)),
    }
    return metrics


def write_outputs(
    out_dir: Path,
    sample_name: str,
    nodes: list[dict],
    edges: list[tuple[int, int, float]],
    track_ids: dict[int, int],
    metrics: dict | None,
    divisions: list[dict] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "nodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "node_id", "track_id", "t", "z", "y", "x", "score"])
        writer.writeheader()
        for node in nodes:
            row = {"dataset": sample_name, "track_id": track_ids[node["node_id"]], **node}
            writer.writerow(row)

    with (out_dir / "edges.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "source_id", "target_id", "distance_um"])
        writer.writeheader()
        for source, target, distance in edges:
            writer.writerow(
                {
                    "dataset": sample_name,
                    "source_id": source,
                    "target_id": target,
                    "distance_um": f"{distance:.4f}",
                }
            )

    if metrics is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if divisions is not None:
        fieldnames = [
            "t",
            "parent_id",
            "daughter_a_id",
            "daughter_b_id",
            "distance_a_um",
            "distance_b_um",
            "daughter_separation_um",
            "midpoint_distance_um",
            "persistence_a",
            "persistence_b",
            "score",
        ]
        with (out_dir / "divisions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for division in divisions:
                writer.writerow({name: division[name] for name in fieldnames})


def main() -> None:
    args = parse_args()
    profile: dict[str, float | int | str] = {}
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    image = open_array(args.zarr_path / args.array_path, mode="r")
    profile["open_seconds"] = time.perf_counter() - stage_start
    sample_name = args.zarr_path.stem
    print(f"sample={sample_name}, shape={image.shape}, dtype={image.dtype}")

    stage_start = time.perf_counter()
    nodes = build_nodes(image, args)
    profile["detect_seconds"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    edges = link_nodes(
        nodes,
        args.link_max_distance_um,
        enable_global_flow=args.enable_global_flow,
        flow_confident_distance_um=args.flow_confident_distance_um,
    )
    profile["link_seconds"] = time.perf_counter() - stage_start

    divisions: list[dict] = []
    if args.enable_divisions:
        stage_start = time.perf_counter()
        edges, divisions = add_division_edges(
            nodes,
            edges,
            max_parent_distance_um=args.division_max_parent_distance_um,
            min_daughter_separation_um=args.division_min_daughter_separation_um,
            max_daughter_separation_um=args.division_max_daughter_separation_um,
            max_midpoint_distance_um=args.division_max_midpoint_distance_um,
            max_nearby_candidates=args.division_max_nearby_candidates,
            persistence_frames=args.division_persistence_frames,
            allow_parent_rewrite=args.division_allow_parent_rewrite,
        )
        profile["division_seconds"] = time.perf_counter() - stage_start
        print(f"division hypotheses accepted: {len(divisions)}")

    stage_start = time.perf_counter()
    track_ids = assign_track_ids(nodes, edges)
    profile["track_assign_seconds"] = time.perf_counter() - stage_start

    metrics = None
    if args.geff is not None:
        stage_start = time.perf_counter()
        gt_nodes, gt_edges = load_geff(args.geff)
        estimated_number_of_nodes = load_estimated_number_of_nodes(args.geff)
        metrics = compute_sparse_metrics(
            nodes,
            edges,
            gt_nodes,
            gt_edges,
            args.link_max_distance_um,
            estimated_number_of_nodes=estimated_number_of_nodes,
        )
        profile["metrics_seconds"] = time.perf_counter() - stage_start
        print(json.dumps(metrics, indent=2))

    stage_start = time.perf_counter()
    write_outputs(args.out_dir, sample_name, nodes, edges, track_ids, metrics, divisions)
    profile["write_seconds"] = time.perf_counter() - stage_start
    profile["total_seconds"] = time.perf_counter() - total_start
    profile["pred_nodes"] = len(nodes)
    profile["pred_edges"] = len(edges)
    profile["sample"] = sample_name
    if args.profile:
        (args.out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(json.dumps(profile, indent=2))
    print(f"wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
