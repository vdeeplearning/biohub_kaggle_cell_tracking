from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_triplet(text: str) -> tuple[int, int, int]:
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated integers, e.g. 5,10,10")
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny 3D CNN to distinguish labeled cell-center patches from conservative background."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--out-path", type=Path, default=Path("/kaggle/working/cell_patch_scorer.pt"))
    parser.add_argument("--metrics-path", type=Path, default=Path("/kaggle/working/patch_scorer_metrics.json"))
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patch-radius-zyx", default="5,10,10")
    parser.add_argument("--max-positives-per-sample", type=int, default=256)
    parser.add_argument("--max-negatives-per-sample", type=int, default=256)
    parser.add_argument("--negative-center-percentile", type=float, default=25.0)
    parser.add_argument("--negative-min-distance-voxels", type=float, default=12.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--install-missing-zarr",
        action="store_true",
        help="Install zarr with pip if Kaggle's image does not already include it.",
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


def crop_patch(volume: np.ndarray, center_zyx: np.ndarray, radius_zyx: tuple[int, int, int]) -> np.ndarray | None:
    center = np.asarray(center_zyx, dtype=np.int64)
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
    patch = np.clip((patch - low) / scale, 0.0, 1.0)
    return patch.astype(np.float16)


def far_from_labels(candidate_zyx: np.ndarray, labels_zyx: np.ndarray, min_distance: float) -> bool:
    if len(labels_zyx) == 0:
        return True
    diff = labels_zyx.astype(np.float32) - candidate_zyx.astype(np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return bool(np.min(dist) >= min_distance)


def sample_negative_centers(
    rng: np.random.Generator,
    volume: np.ndarray,
    labels_zyx: np.ndarray,
    radius_zyx: tuple[int, int, int],
    count: int,
    center_percentile: float,
    min_distance: float,
) -> list[np.ndarray]:
    radius = np.asarray(radius_zyx, dtype=np.int64)
    shape = np.asarray(volume.shape, dtype=np.int64)
    threshold = float(np.percentile(volume, center_percentile))
    centers: list[np.ndarray] = []
    max_tries = max(500, count * 100)
    for _ in range(max_tries):
        if len(centers) >= count:
            break
        center = np.asarray(
            [
                rng.integers(radius[0], shape[0] - radius[0]),
                rng.integers(radius[1], shape[1] - radius[1]),
                rng.integers(radius[2], shape[2] - radius[2]),
            ],
            dtype=np.int64,
        )
        if float(volume[tuple(center)]) > threshold:
            continue
        if not far_from_labels(center, labels_zyx, min_distance=min_distance):
            continue
        centers.append(center)
    return centers


@dataclass
class PatchSet:
    patches: list[np.ndarray]
    labels: list[int]
    samples: list[str]


def build_patch_set(
    zarr_module,
    pairs: list[tuple[str, Path, Path]],
    radius_zyx: tuple[int, int, int],
    max_positives_per_sample: int,
    max_negatives_per_sample: int,
    negative_center_percentile: float,
    negative_min_distance_voxels: float,
    seed: int,
) -> PatchSet:
    rng = np.random.default_rng(seed)
    patches: list[np.ndarray] = []
    labels: list[int] = []
    sample_names: list[str] = []

    for sample_index, (sample_name, zarr_path, geff_path) in enumerate(pairs, start=1):
        image = zarr_module.open(zarr_path / "0", mode="r")
        geff_nodes = load_geff_nodes(zarr_module, geff_path)
        if len(geff_nodes) == 0:
            continue

        order = rng.permutation(len(geff_nodes))
        selected_pos = geff_nodes[order[:max_positives_per_sample]]
        labels_by_t: dict[int, np.ndarray] = {}
        for t in np.unique(geff_nodes[:, 1]):
            labels_by_t[int(t)] = geff_nodes[geff_nodes[:, 1] == t, 2:5]

        positives_added = 0
        negatives_added = 0
        frame_cache: dict[int, np.ndarray] = {}

        for row in selected_pos:
            t = int(row[1])
            if t < 0 or t >= image.shape[0]:
                continue
            if t not in frame_cache:
                frame_cache[t] = np.asarray(image[t, :, :, :])
            patch = crop_patch(frame_cache[t], row[2:5], radius_zyx)
            if patch is None:
                continue
            patches.append(normalize_patch(patch))
            labels.append(1)
            sample_names.append(sample_name)
            positives_added += 1

        time_values = np.unique(geff_nodes[:, 1]).astype(np.int64)
        rng.shuffle(time_values)
        for t in time_values:
            if negatives_added >= max_negatives_per_sample:
                break
            t_int = int(t)
            if t_int < 0 or t_int >= image.shape[0]:
                continue
            if t_int not in frame_cache:
                frame_cache[t_int] = np.asarray(image[t_int, :, :, :])
            volume = frame_cache[t_int]
            needed = max_negatives_per_sample - negatives_added
            centers = sample_negative_centers(
                rng,
                volume,
                labels_by_t.get(t_int, np.zeros((0, 3), dtype=np.int64)),
                radius_zyx,
                count=min(needed, 64),
                center_percentile=negative_center_percentile,
                min_distance=negative_min_distance_voxels,
            )
            for center in centers:
                patch = crop_patch(volume, center, radius_zyx)
                if patch is None:
                    continue
                patches.append(normalize_patch(patch))
                labels.append(0)
                sample_names.append(sample_name)
                negatives_added += 1
                if negatives_added >= max_negatives_per_sample:
                    break

        print(
            f"[{sample_index}/{len(pairs)}] {sample_name}: "
            f"positives={positives_added}, negatives={negatives_added}",
            flush=True,
        )

    return PatchSet(patches=patches, labels=labels, samples=sample_names)


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
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            probs = torch.sigmoid(logits)
            pred = probs >= 0.5
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
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "precision": precision,
        "recall": recall,
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
    radius_zyx = parse_triplet(args.patch_radius_zyx)
    patch_shape = tuple(2 * r + 1 for r in radius_zyx)

    print(f"using {len(pairs)} samples from {args.train_dir}")
    print(f"patch shape zyx={patch_shape}")
    patch_set = build_patch_set(
        zarr_module,
        pairs,
        radius_zyx=radius_zyx,
        max_positives_per_sample=args.max_positives_per_sample,
        max_negatives_per_sample=args.max_negatives_per_sample,
        negative_center_percentile=args.negative_center_percentile,
        negative_min_distance_voxels=args.negative_min_distance_voxels,
        seed=args.seed,
    )
    if not patch_set.patches:
        raise SystemExit("No patches were extracted.")

    train_indices, val_indices, val_samples = split_by_sample(patch_set.samples, args.val_fraction, args.seed)
    if len(train_indices) == 0 or len(val_indices) == 0:
        raise SystemExit("Train/validation split is empty; use more samples or lower --val-fraction.")

    train_ds = PatchDataset(patch_set.patches, patch_set.labels, train_indices)
    val_ds = PatchDataset(patch_set.patches, patch_set.labels, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyPatchScorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []

    print(f"device={device}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"train patches={len(train_ds)}, val patches={len(val_ds)}, val samples={val_samples}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss_running": total_loss / max(1, total),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_precision={val_metrics['precision']:.3f} val_recall={val_metrics['recall']:.3f}",
            flush=True,
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "patch_radius_zyx": radius_zyx,
            "patch_shape_zyx": patch_shape,
            "model": "TinyPatchScorer",
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
        "train_patches": int(len(train_ds)),
        "val_patches": int(len(val_ds)),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "history": history,
        "out_path": str(args.out_path),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote weights: {args.out_path}")
    print(f"wrote metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
