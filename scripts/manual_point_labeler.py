from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


VOXEL_SCALE_TZYX = (1.0, 1.625, 0.40625, 0.40625)
LABEL_SPECS = {
    "positive": {"face_color": "cyan", "size": 9.0},
    "negative": {"face_color": "red", "size": 8.0},
    "ignore": {"face_color": "yellow", "size": 8.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-slice napari labeler for manual Biohub centroid labels. "
            "Use 1/2/3 to select positive/negative/ignore, click points, N for next slice, S to save."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/train"),
        help="Directory containing training .zarr samples.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("manual_labels/manual_centroids.csv"),
        help="CSV file where manual point labels are saved.",
    )
    parser.add_argument(
        "--array-path",
        default="0",
        help="Array path inside each .zarr store. Competition samples use 0.",
    )
    parser.add_argument(
        "--sample",
        default="",
        help="Optional sample stem to focus on, e.g. 44b6_0b24845f.",
    )
    parser.add_argument("--contrast-min", type=float, default=0.0)
    parser.add_argument("--contrast-max", type=float, default=3500.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def find_zarrs(train_dir: Path, sample: str) -> list[Path]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Could not find train directory: {train_dir}")

    if sample:
        path = train_dir / f"{sample}.zarr"
        if not path.exists():
            raise FileNotFoundError(f"Could not find requested sample: {path}")
        return [path]

    zarrs = sorted(train_dir.glob("*.zarr"))
    if not zarrs:
        raise FileNotFoundError(f"No .zarr samples found in {train_dir}")
    return zarrs


def open_image(zarr_path: Path, array_path: str) -> da.Array:
    array_root = zarr_path / array_path
    if not array_root.exists():
        raise FileNotFoundError(f"Could not find array path: {array_root}")
    return da.from_zarr(zarr.open(array_root, mode="r"))


def read_all_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_all_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "t", "z", "y", "x", "label", "source"]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def rows_for_sample(path: Path, sample: str) -> dict[str, np.ndarray]:
    points_by_label: dict[str, list[list[float]]] = {label: [] for label in LABEL_SPECS}
    for row in read_all_rows(path):
        if row.get("dataset") != sample:
            continue
        label = row.get("label", "")
        if label not in points_by_label:
            continue
        points_by_label[label].append(
            [
                float(row["t"]),
                float(row["z"]),
                float(row["y"]),
                float(row["x"]),
            ]
        )
    return {
        label: np.asarray(points, dtype=float).reshape((-1, 4))
        for label, points in points_by_label.items()
    }


def format_coord(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}"


class ManualPointLabeler:
    def __init__(
        self,
        viewer: napari.Viewer,
        zarrs: list[Path],
        output_csv: Path,
        array_path: str,
        contrast_limits: tuple[float, float],
    ) -> None:
        self.viewer = viewer
        self.zarrs = zarrs
        self.output_csv = output_csv
        self.array_path = array_path
        self.contrast_limits = contrast_limits
        self.current_zarr: Path | None = None
        self.current_sample = ""
        self.current_shape: tuple[int, int, int, int] | None = None
        self.status = QLabel("")
        self.short_help = QLabel(
            "1 positive | 2 negative | 3 ignore\n"
            "Click to add points. Select a point and press Delete to remove.\n"
            "N next random slice | S save"
        )
        self.label_layers: dict[str, napari.layers.Points] = {}
        self._loading = False

    def add_controls(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout()

        positive_button = QPushButton("Positive (1)")
        negative_button = QPushButton("Negative (2)")
        ignore_button = QPushButton("Ignore (3)")
        next_button = QPushButton("Next random slice (N)")
        save_button = QPushButton("Save (S)")

        positive_button.clicked.connect(lambda: self.select_label("positive"))
        negative_button.clicked.connect(lambda: self.select_label("negative"))
        ignore_button.clicked.connect(lambda: self.select_label("ignore"))
        next_button.clicked.connect(self.next_random_slice)
        save_button.clicked.connect(self.save_current_sample)

        layout.addWidget(self.short_help)
        layout.addWidget(positive_button)
        layout.addWidget(negative_button)
        layout.addWidget(ignore_button)
        layout.addWidget(next_button)
        layout.addWidget(save_button)
        layout.addWidget(self.status)
        widget.setLayout(layout)
        self.viewer.window.add_dock_widget(widget, area="right", name="Manual labels")

    def bind_keys(self) -> None:
        @self.viewer.bind_key("1", overwrite=True)
        def _select_positive(viewer: napari.Viewer) -> None:
            self.select_label("positive")

        @self.viewer.bind_key("2", overwrite=True)
        def _select_negative(viewer: napari.Viewer) -> None:
            self.select_label("negative")

        @self.viewer.bind_key("3", overwrite=True)
        def _select_ignore(viewer: napari.Viewer) -> None:
            self.select_label("ignore")

        @self.viewer.bind_key("n", overwrite=True)
        def _next(viewer: napari.Viewer) -> None:
            self.next_random_slice()

        @self.viewer.bind_key("s", overwrite=True)
        def _save(viewer: napari.Viewer) -> None:
            self.save_current_sample()

    def next_random_slice(self) -> None:
        if self.current_zarr is not None:
            self.save_current_sample()

        zarr_path = random.choice(self.zarrs)
        image = open_image(zarr_path, self.array_path)
        if len(image.shape) != 4:
            raise ValueError(f"Expected TZYX image, got shape {image.shape} for {zarr_path}")

        t_index = random.randrange(image.shape[0])
        z_index = random.randrange(image.shape[1])
        self.load_sample(zarr_path, image, t_index, z_index)

    def load_sample(self, zarr_path: Path, image: da.Array, t_index: int, z_index: int) -> None:
        self._loading = True
        self.viewer.layers.clear()
        self.current_zarr = zarr_path
        self.current_sample = zarr_path.stem
        self.current_shape = tuple(int(v) for v in image.shape)

        self.viewer.add_image(
            image,
            name=self.current_sample,
            scale=VOXEL_SCALE_TZYX,
            blending="additive",
            contrast_limits=self.contrast_limits,
        )

        existing = rows_for_sample(self.output_csv, self.current_sample)
        self.label_layers = {}
        for label, spec in LABEL_SPECS.items():
            layer = self.viewer.add_points(
                existing[label],
                ndim=4,
                name=label,
                scale=VOXEL_SCALE_TZYX,
                size=spec["size"],
                face_color=spec["face_color"],
                opacity=0.95,
            )
            layer.mode = "add"
            layer.events.data.connect(self.autosave_current_sample)
            self.label_layers[label] = layer

        try:
            self.viewer.dims.axis_labels = ("t", "z", "y", "x")
            self.viewer.dims.current_step = (t_index, z_index, 0, 0)
        except Exception:
            pass

        self.select_label("positive")
        self._loading = False
        self.update_status()

    def select_label(self, label: str) -> None:
        layer = self.label_layers.get(label)
        if layer is None:
            return
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self.update_status(active_label=label)

    def rows_from_current_layers(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for label, layer in self.label_layers.items():
            data = np.asarray(layer.data, dtype=float).reshape((-1, 4))
            for t, z, y, x in data:
                rows.append(
                    {
                        "dataset": self.current_sample,
                        "t": format_coord(t),
                        "z": format_coord(z),
                        "y": format_coord(y),
                        "x": format_coord(x),
                        "label": label,
                        "source": "manual",
                    }
                )
        return rows

    def save_current_sample(self) -> None:
        if self._loading or not self.current_sample:
            return
        other_rows = [
            row
            for row in read_all_rows(self.output_csv)
            if row.get("dataset") != self.current_sample
        ]
        rows = other_rows + self.rows_from_current_layers()
        write_all_rows(self.output_csv, rows)
        self.update_status(saved=True)

    def autosave_current_sample(self, event=None) -> None:
        if not self._loading:
            self.save_current_sample()

    def update_status(self, active_label: str | None = None, saved: bool = False) -> None:
        counts = {
            label: len(np.asarray(layer.data).reshape((-1, 4)))
            for label, layer in self.label_layers.items()
        }
        active = active_label
        if active is None and self.viewer.layers.selection.active is not None:
            active = self.viewer.layers.selection.active.name
        tzyx = tuple(int(v) for v in self.viewer.dims.current_step)
        prefix = "Saved. " if saved else ""
        self.status.setText(
            f"{prefix}{self.current_sample}\n"
            f"slice t={tzyx[0]}, z={tzyx[1]}\n"
            f"active={active}\n"
            f"positive={counts.get('positive', 0)} "
            f"negative={counts.get('negative', 0)} "
            f"ignore={counts.get('ignore', 0)}\n"
            f"{self.output_csv}"
        )


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    zarrs = find_zarrs(args.train_dir, args.sample)
    print(f"Found {len(zarrs)} zarr sample(s) in {args.train_dir}")
    print(f"Saving manual labels to {args.output_csv}")

    viewer = napari.Viewer()
    labeler = ManualPointLabeler(
        viewer=viewer,
        zarrs=zarrs,
        output_csv=args.output_csv,
        array_path=args.array_path,
        contrast_limits=(args.contrast_min, args.contrast_max),
    )
    labeler.add_controls()
    labeler.bind_keys()
    labeler.next_random_slice()
    napari.run()


if __name__ == "__main__":
    main()
