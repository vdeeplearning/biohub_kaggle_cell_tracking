from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCALE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


def parse_triplet(text: str, dtype=float):
    parts = [dtype(v.strip()) for v in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated values, e.g. 5,10,10")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny 3D CNN to rescore classical cell-center proposals."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--out-path", type=Path, default=Path("/kaggle/working/cell_proposal_scorer.pt"))
    parser.add_argument("--metrics-path", type=Path, default=Path("/kaggle/working/proposal_scorer_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-radius-zyx", default="5,10,10")
    parser.add_argument("--gaussian-sigma-zyx", default="0.7,1.0,1.0")
    parser.add_argument("--threshold-quantile", type=float, default=50.0)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--max-proposals-per-frame", type=int, default=900)
    parser.add_argument("--max-positive-proposals-per-sample", type=int, default=384)
    parser.add_argument("--max-negative-proposals-per-sample", type=int, default=384)
    parser.add_argument("--positive-distance-um", type=float, default=4.0)
    parser.add_argument("--negative-distance-um", type=float, default=10.0)
    parser.add_argument(
        "--negative-score-quantile",
        type=float,
        default=35.0,
        help="Use only low-scoring unmatched proposals as conservative negatives.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--install-missing-zarr", action="store_true")
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=Path("manual_labels/manual_centroids.csv"),
        help=(
            "Optional CSV of manual point labels with columns dataset,t,z,y,x,label,source. "
            "Positive and negative rows are cropped directly as extra training examples."
        ),
    )
    parser.add_argument(
        "--manual-repeat",
        type=int,
        default=1,
        help="Duplicate manual positive/negative patches this many times to upweight them.",
    )
    return parser.parse_args()


def import_zarr(install_missing: bool):
    try:
        import zarr  # type: ignore

        return zarr
    except ModuleNotFoundError:
        if not install_missing:
            raise ModuleNotFoundError(
                "zarr is not installed. In Kaggle, rerun with --install-missing-zarr "
                "or run `!pip install -q zarr` first."
            )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=True)
        import zarr  # type: ignore

        return zarr


def find_pairs(train_dir: Path) -> list[tuple[str, Path, Path]]:
    zarrs = {p.name.removesuffix(".zarr"): p for p in train_dir.glob("*.zarr") if p.is_dir()}
    geffs = {p.name.removesuffix(".geff"): p for p in train_dir.glob("*.geff") if p.is_dir()}
    return [(name, zarrs[name], geffs[name]) for name in sorted(set(zarrs) & set(geffs))]


def load_geff_nodes(zarr_module, geff_path: Path) -> np.ndarray:
    root = zarr_module.open_group(geff_path, mode="r")
    node_ids = np.asarray(root["nodes/ids"]).astype(np.int64)
    t = np.asarray(root["nodes/props/t/values"]).astype(np.int64)
    z = np.asarray(root["nodes/props/z/values"]).astype(np.int64)
    y = np.asarray(root["nodes/props/y/values"]).astype(np.int64)
    x = np.asarray(root["nodes/props/x/values"]).astype(np.int64)
    return np.stack([node_ids, t, z, y, x], axis=1)


def load_manual_labels(path: Path) -> dict[str, list[tuple[int, np.ndarray, int]]]:
    labels_by_sample: dict[str, list[tuple[int, np.ndarray, int]]] = {}
    if not path.exists():
        return labels_by_sample

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_text = row.get("label", "").strip().lower()
            if label_text == "positive":
                label = 1
            elif label_text == "negative":
                label = 0
            else:
                continue

            sample_name = row.get("dataset", "").strip()
            if not sample_name:
                continue
            t = int(round(float(row["t"])))
            zyx = np.asarray(
                [
                    float(row["z"]),
                    float(row["y"]),
                    float(row["x"]),
                ],
                dtype=np.float32,
            )
            labels_by_sample.setdefault(sample_name, []).append((t, zyx, label))
    return labels_by_sample


def crop_patch(volume: np.ndarray, center_zyx: np.ndarray, radius_zyx: tuple[int, int, int]) -> np.ndarray | None:
    center = np.asarray(np.rint(center_zyx), dtype=np.int64)
    radius = np.asarray(radius_zyx, dtype=np.int64)
    lo = center - radius
    hi = center + radius + 1
    shape = np.asarray(volume.shape, dtype=np.int64)
    if np.any(lo < 0) or np.any(hi > shape):
        return None
    return volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32, copy=False)
    low = float(np.percentile(patch, 10))
    high = float(np.percentile(patch, 99))
    scale = max(high - low, 1.0)
    return np.clip((patch - low) / scale, 0.0, 1.0).astype(np.float16)


def detect_proposals(
    volume: np.ndarray,
    sigma_zyx: tuple[float, float, float],
    threshold_quantile: float,
    min_peak_distance: int,
    max_proposals: int,
) -> tuple[np.ndarray, np.ndarray]:
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma_zyx)
    positive = smoothed[smoothed > 0]
    if len(positive) == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    threshold = float(np.percentile(positive, threshold_quantile))
    peaks = peak_local_max(
        smoothed,
        min_distance=min_peak_distance,
        threshold_abs=threshold,
        exclude_border=False,
        num_peaks=max_proposals,
    )
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    scores = smoothed[peaks[:, 0], peaks[:, 1], peaks[:, 2]].astype(np.float32)
    order = np.argsort(-scores)
    return peaks[order], scores[order]


def nearest_label_distances_um(proposals_zyx: np.ndarray, labels_zyx: np.ndarray) -> np.ndarray:
    if len(proposals_zyx) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(labels_zyx) == 0:
        return np.full((len(proposals_zyx),), np.inf, dtype=np.float32)
    proposal_um = proposals_zyx.astype(np.float32) * SCALE_ZYX
    label_um = labels_zyx.astype(np.float32) * SCALE_ZYX
    diff = proposal_um[:, None, :] - label_um[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1).astype(np.float32)


@dataclass
class ProposalPatchSet:
    patches: list[np.ndarray]
    labels: list[int]
    samples: list[str]
    positive_candidates: int
    negative_candidates: int
    ignored_candidates: int
    manual_positive_patches: int
    manual_negative_patches: int
    manual_skipped_labels: int


def build_patch_set(
    zarr_module,
    pairs: list[tuple[str, Path, Path]],
    radius_zyx: tuple[int, int, int],
    sigma_zyx: tuple[float, float, float],
    threshold_quantile: float,
    min_peak_distance: int,
    max_proposals_per_frame: int,
    positive_distance_um: float,
    negative_distance_um: float,
    negative_score_quantile: float,
    max_positive_proposals_per_sample: int,
    max_negative_proposals_per_sample: int,
    manual_labels_by_sample: dict[str, list[tuple[int, np.ndarray, int]]],
    manual_repeat: int,
    seed: int,
) -> ProposalPatchSet:
    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    labels: list[int] = []
    sample_names: list[str] = []
    total_positive_candidates = 0
    total_negative_candidates = 0
    total_ignored_candidates = 0
    total_manual_positive = 0
    total_manual_negative = 0
    total_manual_skipped = 0

    for sample_index, (sample_name, zarr_path, geff_path) in enumerate(pairs, start=1):
        image = zarr_module.open(zarr_path / "0", mode="r")
        geff_nodes = load_geff_nodes(zarr_module, geff_path)
        labels_by_t = {int(t): geff_nodes[geff_nodes[:, 1] == t, 2:5] for t in np.unique(geff_nodes[:, 1])}

        pos_pool: list[tuple[np.ndarray, int]] = []
        neg_pool: list[tuple[np.ndarray, int]] = []
        ignored = 0
        for t in sorted(labels_by_t):
            if t < 0 or t >= image.shape[0]:
                continue
            volume = np.asarray(image[t, :, :, :])
            proposals, scores = detect_proposals(
                volume,
                sigma_zyx=sigma_zyx,
                threshold_quantile=threshold_quantile,
                min_peak_distance=min_peak_distance,
                max_proposals=max_proposals_per_frame,
            )
            if len(proposals) == 0:
                continue
            dists = nearest_label_distances_um(proposals, labels_by_t[t])
            positive_mask = dists <= positive_distance_um
            far_mask = dists >= negative_distance_um
            low_score_cutoff = float(np.percentile(scores, negative_score_quantile))
            negative_mask = far_mask & (scores <= low_score_cutoff)
            ignored += int(len(proposals) - np.count_nonzero(positive_mask) - np.count_nonzero(negative_mask))
            pos_pool.extend((proposal, t) for proposal in proposals[positive_mask])
            neg_pool.extend((proposal, t) for proposal in proposals[negative_mask])

        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)
        selected_pos = pos_pool[:max_positive_proposals_per_sample]
        selected_neg = neg_pool[:max_negative_proposals_per_sample]
        frame_cache: dict[int, np.ndarray] = {}

        added_pos = 0
        added_neg = 0
        for pool, label in [(selected_pos, 1), (selected_neg, 0)]:
            for proposal, t in pool:
                if t not in frame_cache:
                    frame_cache[t] = np.asarray(image[t, :, :, :])
                patch = crop_patch(frame_cache[t], proposal, radius_zyx)
                if patch is None:
                    continue
                patches.append(normalize_patch(patch))
                labels.append(label)
                sample_names.append(sample_name)
                if label == 1:
                    added_pos += 1
                else:
                    added_neg += 1

        manual_added_pos = 0
        manual_added_neg = 0
        sample_manual_labels = manual_labels_by_sample.get(sample_name, [])
        for t, zyx, label in sample_manual_labels:
            if t < 0 or t >= image.shape[0]:
                total_manual_skipped += 1
                continue
            if t not in frame_cache:
                frame_cache[t] = np.asarray(image[t, :, :, :])
            patch = crop_patch(frame_cache[t], zyx, radius_zyx)
            if patch is None:
                total_manual_skipped += 1
                continue
            normalized = normalize_patch(patch)
            for _ in range(max(1, manual_repeat)):
                patches.append(normalized)
                labels.append(label)
                sample_names.append(sample_name)
            if label == 1:
                manual_added_pos += max(1, manual_repeat)
            else:
                manual_added_neg += max(1, manual_repeat)

        total_positive_candidates += len(pos_pool)
        total_negative_candidates += len(neg_pool)
        total_ignored_candidates += ignored
        total_manual_positive += manual_added_pos
        total_manual_negative += manual_added_neg
        print(
            f"[{sample_index}/{len(pairs)}] {sample_name}: "
            f"pos_candidates={len(pos_pool)}, neg_candidates={len(neg_pool)}, ignored={ignored}, "
            f"added_pos={added_pos}, added_neg={added_neg}, "
            f"manual_pos={manual_added_pos}, manual_neg={manual_added_neg}",
            flush=True,
        )

    selected_names = {name for name, _, _ in pairs}
    total_manual_skipped += sum(
        len(rows) for name, rows in manual_labels_by_sample.items() if name not in selected_names
    )

    return ProposalPatchSet(
        patches=patches,
        labels=labels,
        samples=sample_names,
        positive_candidates=total_positive_candidates,
        negative_candidates=total_negative_candidates,
        ignored_candidates=total_ignored_candidates,
        manual_positive_patches=total_manual_positive,
        manual_negative_patches=total_manual_negative,
        manual_skipped_labels=total_manual_skipped,
    )


class PatchDataset(Dataset):
    def __init__(self, patches: list[np.ndarray], labels: list[int], indices: np.ndarray):
        self.patches = patches
        self.labels = labels
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        source_idx = int(self.indices[idx])
        patch = torch.from_numpy(self.patches[source_idx].astype(np.float32, copy=False))[None, :, :, :]
        label = torch.tensor(float(self.labels[source_idx]), dtype=torch.float32)
        return patch, label


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(1)


def split_by_sample(samples: list[str], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    unique_samples = sorted(set(samples))
    rng = random.Random(seed)
    rng.shuffle(unique_samples)
    val_count = max(1, int(round(len(unique_samples) * val_fraction)))
    val_samples = set(unique_samples[:val_count])
    train_indices = np.asarray([i for i, name in enumerate(samples) if name not in val_samples], dtype=np.int64)
    val_indices = np.asarray([i for i, name in enumerate(samples) if name in val_samples], dtype=np.int64)
    return train_indices, val_indices, sorted(val_samples)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total = correct = tp = fp = tn = fn = 0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            pred = torch.sigmoid(logits) >= 0.5
            truth = y >= 0.5
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((pred == truth).sum().item())
            tp += int((pred & truth).sum().item())
            fp += int((pred & ~truth).sum().item())
            tn += int((~pred & ~truth).sum().item())
            fn += int((~pred & truth).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": total,
    }


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    zarr_module = import_zarr(args.install_missing_zarr)
    pairs = find_pairs(args.train_dir)
    if not pairs:
        raise SystemExit(f"No .zarr/.geff pairs found in {args.train_dir}")
    pairs = pairs[: args.max_samples]
    radius_zyx = parse_triplet(args.patch_radius_zyx, int)
    sigma_zyx = parse_triplet(args.gaussian_sigma_zyx, float)
    patch_shape = tuple(2 * r + 1 for r in radius_zyx)

    print(f"using {len(pairs)} samples from {args.train_dir}")
    print(f"patch shape zyx={patch_shape}")
    manual_labels_by_sample = load_manual_labels(args.manual_labels)
    manual_label_count = sum(len(rows) for rows in manual_labels_by_sample.values())
    if manual_label_count:
        print(f"using {manual_label_count} manual labels from {args.manual_labels}")
    else:
        print(f"no manual positive/negative labels found at {args.manual_labels}")
    patch_set = build_patch_set(
        zarr_module,
        pairs,
        radius_zyx=radius_zyx,
        sigma_zyx=sigma_zyx,
        threshold_quantile=args.threshold_quantile,
        min_peak_distance=args.min_peak_distance,
        max_proposals_per_frame=args.max_proposals_per_frame,
        positive_distance_um=args.positive_distance_um,
        negative_distance_um=args.negative_distance_um,
        negative_score_quantile=args.negative_score_quantile,
        max_positive_proposals_per_sample=args.max_positive_proposals_per_sample,
        max_negative_proposals_per_sample=args.max_negative_proposals_per_sample,
        manual_labels_by_sample=manual_labels_by_sample,
        manual_repeat=args.manual_repeat,
        seed=args.seed,
    )
    if not patch_set.patches:
        raise SystemExit("No proposal patches were extracted.")

    train_indices, val_indices, val_samples = split_by_sample(patch_set.samples, args.val_fraction, args.seed)
    train_ds = PatchDataset(patch_set.patches, patch_set.labels, train_indices)
    val_ds = PatchDataset(patch_set.patches, patch_set.labels, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPatchScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_epoch = 0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_state_dict = None

    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"train patches={len(train_ds)}, val patches={len(val_ds)}, val samples={val_samples}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.numel())
            running_total += int(y.numel())

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss_running": running_loss / max(1, running_total),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        is_better = (
            val_metrics["f1"] > best_val_f1
            or (val_metrics["f1"] == best_val_f1 and val_metrics["loss"] < best_val_loss)
        )
        if is_better:
            best_epoch = epoch
            best_val_f1 = float(val_metrics["f1"])
            best_val_loss = float(val_metrics["loss"])
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_precision={val_metrics['precision']:.3f} val_recall={val_metrics['recall']:.3f} "
            f"val_f1={val_metrics['f1']:.3f}",
            flush=True,
        )

    if best_state_dict is None:
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "patch_radius_zyx": radius_zyx,
            "patch_shape_zyx": patch_shape,
            "model": "TinyPatchScorer",
            "training_mode": "classical_proposal_rescorer",
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
        },
        args.out_path,
    )

    label_array = np.asarray(patch_set.labels, dtype=np.int64)
    metrics = {
        "seconds": time.perf_counter() - start,
        "train_dir": str(args.train_dir),
        "samples_used": [name for name, _, _ in pairs],
        "val_samples": val_samples,
        "patch_radius_zyx": radius_zyx,
        "patch_shape_zyx": patch_shape,
        "total_patches": len(patch_set.patches),
        "positive_patches": int(np.sum(label_array == 1)),
        "negative_patches": int(np.sum(label_array == 0)),
        "positive_candidates_before_cap": patch_set.positive_candidates,
        "negative_candidates_before_cap": patch_set.negative_candidates,
        "ignored_candidates": patch_set.ignored_candidates,
        "manual_labels_path": str(args.manual_labels),
        "manual_label_rows_loaded": manual_label_count,
        "manual_positive_patches": patch_set.manual_positive_patches,
        "manual_negative_patches": patch_set.manual_negative_patches,
        "manual_skipped_labels": patch_set.manual_skipped_labels,
        "manual_repeat": args.manual_repeat,
        "train_patches": int(len(train_ds)),
        "val_patches": int(len(val_ds)),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "history": history,
        "out_path": str(args.out_path),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote best epoch {best_epoch} weights: {args.out_path}")
    print(f"wrote metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
