from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np
import zarr
from scipy.spatial import cKDTree
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


SCALE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float32)


@dataclass
class Node:
    node_id: int
    track_id: int
    t: int
    zyx: np.ndarray
    score: float


@dataclass
class DivisionCandidate:
    dataset: str
    parent: Node
    daughter_a: Node
    daughter_b: Node
    score: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XY max-projection labeler for candidate cell division triplets."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to one .zarr sample.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory containing classical nodes.csv/edges.csv for candidate mining.",
    )
    parser.add_argument(
        "--geff",
        type=Path,
        default=None,
        help="Optional matching .geff. Known sparse divisions are added as positive-reference candidates.",
    )
    parser.add_argument("--array-path", default="0")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("manual_labels/manual_divisions.csv"),
    )
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--time-before", type=int, default=2)
    parser.add_argument("--time-after", type=int, default=4)
    parser.add_argument("--z-radius", type=int, default=4)
    parser.add_argument("--xy-radius", type=int, default=48)
    parser.add_argument("--contrast-min", type=float, default=0.0)
    parser.add_argument("--contrast-max", type=float, default=3500.0)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--max-parent-distance-um", type=float, default=10.0)
    parser.add_argument("--min-daughter-separation-um", type=float, default=2.0)
    parser.add_argument("--max-daughter-separation-um", type=float, default=15.0)
    parser.add_argument("--max-midpoint-distance-um", type=float, default=5.0)
    parser.add_argument("--max-nearby-candidates", type=int, default=5)
    return parser.parse_args()


def read_nodes(path: Path) -> list[Node]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                Node(
                    node_id=int(row["node_id"]),
                    track_id=int(row.get("track_id", row["node_id"])),
                    t=int(row["t"]),
                    zyx=np.asarray([float(row["z"]), float(row["y"]), float(row["x"])], dtype=np.float32),
                    score=float(row.get("score", 1.0)),
                )
            )
        return rows


def read_edges(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [(int(row["source_id"]), int(row["target_id"])) for row in csv.DictReader(f)]


def distance_um(a_zyx: np.ndarray, b_zyx: np.ndarray) -> float:
    delta = (a_zyx.astype(np.float32) - b_zyx.astype(np.float32)) * SCALE_ZYX
    return float(np.sqrt(np.sum(delta * delta)))


def mine_candidates(result_dir: Path, dataset: str, max_candidates: int, args: argparse.Namespace) -> list[DivisionCandidate]:
    nodes = read_nodes(result_dir / "nodes.csv")
    if not nodes:
        return []
    nodes_by_t: dict[int, list[Node]] = {}
    by_id = {}
    for node in nodes:
        nodes_by_t.setdefault(node.t, []).append(node)
        by_id[node.node_id] = node

    outgoing: dict[int, list[int]] = {}
    for source_id, target_id in read_edges(result_dir / "edges.csv"):
        outgoing.setdefault(source_id, []).append(target_id)

    candidates: list[DivisionCandidate] = []
    for t in sorted(nodes_by_t):
        next_nodes = nodes_by_t.get(t + 1, [])
        if len(next_nodes) < 2:
            continue
        next_points_um = np.asarray([node.zyx * SCALE_ZYX for node in next_nodes], dtype=np.float32)
        next_tree = cKDTree(next_points_um)
        for parent in nodes_by_t[t]:
            parent_um = parent.zyx * SCALE_ZYX
            nearby_indices = next_tree.query_ball_point(parent_um, r=args.max_parent_distance_um)
            nearby = [next_nodes[i] for i in nearby_indices]
            if len(nearby) < 2:
                continue
            nearby.sort(key=lambda child: distance_um(parent.zyx, child.zyx))
            nearby = nearby[: args.max_nearby_candidates]

            for i in range(len(nearby)):
                for j in range(i + 1, len(nearby)):
                    a = nearby[i]
                    b = nearby[j]
                    daughter_sep = distance_um(a.zyx, b.zyx)
                    if daughter_sep < args.min_daughter_separation_um or daughter_sep > args.max_daughter_separation_um:
                        continue
                    midpoint = (a.zyx + b.zyx) * 0.5
                    midpoint_distance = distance_um(parent.zyx, midpoint)
                    if midpoint_distance > args.max_midpoint_distance_um:
                        continue

                    linked_targets = set(outgoing.get(parent.node_id, []))
                    existing_link_bonus = 0.5 * int(a.node_id in linked_targets) + 0.5 * int(b.node_id in linked_targets)
                    score = (
                        3.0 / max(0.5, midpoint_distance)
                        + 1.0 / max(0.5, abs(daughter_sep - 7.0))
                        + 0.001 * (a.score + b.score)
                        + existing_link_bonus
                    )
                    candidates.append(DivisionCandidate(dataset, parent, a, b, score, "mined"))

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    deduped = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.parent.node_id,
            min(candidate.daughter_a.node_id, candidate.daughter_b.node_id),
            max(candidate.daughter_a.node_id, candidate.daughter_b.node_id),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break
    return deduped


def load_geff_divisions(geff_path: Path, dataset: str) -> list[DivisionCandidate]:
    root = zarr.open_group(geff_path, mode="r")
    node_ids = np.asarray(root["nodes/ids"]).astype(np.int64)
    t = np.asarray(root["nodes/props/t/values"]).astype(np.int64)
    z = np.asarray(root["nodes/props/z/values"]).astype(np.int64)
    y = np.asarray(root["nodes/props/y/values"]).astype(np.int64)
    x = np.asarray(root["nodes/props/x/values"]).astype(np.int64)
    edges = np.asarray(root["edges/ids"]).astype(np.int64)
    if edges.ndim == 1 and len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int64)

    by_id = {
        int(node_id): Node(
            node_id=int(node_id),
            track_id=int(node_id),
            t=int(t_i),
            zyx=np.asarray([z_i, y_i, x_i], dtype=np.float32),
            score=1.0,
        )
        for node_id, t_i, z_i, y_i, x_i in zip(node_ids, t, z, y, x)
    }
    targets: dict[int, list[int]] = {}
    for source_id, target_id in edges:
        targets.setdefault(int(source_id), []).append(int(target_id))

    candidates = []
    for source_id, target_ids in sorted(targets.items()):
        valid = [target_id for target_id in target_ids if target_id in by_id]
        if len(valid) < 2 or source_id not in by_id:
            continue
        a, b = sorted([by_id[target_id] for target_id in valid], key=lambda node: node.node_id)[:2]
        candidates.append(DivisionCandidate(dataset, by_id[source_id], a, b, 9999.0, "geff_true"))
    return candidates


def read_existing_labels(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "label",
        "source",
        "candidate_score",
        "parent_node_id",
        "parent_t",
        "parent_z",
        "parent_y",
        "parent_x",
        "daughter_a_node_id",
        "daughter_a_t",
        "daughter_a_z",
        "daughter_a_y",
        "daughter_a_x",
        "daughter_b_node_id",
        "daughter_b_t",
        "daughter_b_z",
        "daughter_b_y",
        "daughter_b_x",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def candidate_key(candidate: DivisionCandidate) -> tuple[str, int, int, int]:
    return (
        candidate.dataset,
        candidate.parent.node_id,
        min(candidate.daughter_a.node_id, candidate.daughter_b.node_id),
        max(candidate.daughter_a.node_id, candidate.daughter_b.node_id),
    )


def make_label_row(candidate: DivisionCandidate, label: str) -> dict[str, str]:
    p = candidate.parent
    a = candidate.daughter_a
    b = candidate.daughter_b
    return {
        "dataset": candidate.dataset,
        "label": label,
        "source": candidate.source,
        "candidate_score": f"{candidate.score:.6f}",
        "parent_node_id": str(p.node_id),
        "parent_t": str(p.t),
        "parent_z": f"{p.zyx[0]:.3f}",
        "parent_y": f"{p.zyx[1]:.3f}",
        "parent_x": f"{p.zyx[2]:.3f}",
        "daughter_a_node_id": str(a.node_id),
        "daughter_a_t": str(a.t),
        "daughter_a_z": f"{a.zyx[0]:.3f}",
        "daughter_a_y": f"{a.zyx[1]:.3f}",
        "daughter_a_x": f"{a.zyx[2]:.3f}",
        "daughter_b_node_id": str(b.node_id),
        "daughter_b_t": str(b.t),
        "daughter_b_z": f"{b.zyx[0]:.3f}",
        "daughter_b_y": f"{b.zyx[1]:.3f}",
        "daughter_b_x": f"{b.zyx[2]:.3f}",
    }


def crop_project_movie(image, candidate: DivisionCandidate, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [candidate.parent.t, *candidate.parent.zyx],
            [candidate.daughter_a.t, *candidate.daughter_a.zyx],
            [candidate.daughter_b.t, *candidate.daughter_b.zyx],
        ],
        dtype=float,
    )
    center = np.rint(points.mean(axis=0)).astype(int)
    t0 = max(0, candidate.parent.t - args.time_before)
    t1 = min(image.shape[0], candidate.parent.t + args.time_after + 1)
    z0 = max(0, int(center[1]) - args.z_radius)
    z1 = min(image.shape[1], int(center[1]) + args.z_radius + 1)
    y0 = max(0, int(center[2]) - args.xy_radius)
    y1 = min(image.shape[2], int(center[2]) + args.xy_radius + 1)
    x0 = max(0, int(center[3]) - args.xy_radius)
    x1 = min(image.shape[3], int(center[3]) + args.xy_radius + 1)

    # Include all three annotated coordinates even if the mean-centered crop is tight.
    t0 = min(t0, int(np.floor(points[:, 0].min())))
    t1 = max(t1, int(np.ceil(points[:, 0].max())) + 1)
    z0 = min(z0, int(np.floor(points[:, 1].min())))
    z1 = max(z1, int(np.ceil(points[:, 1].max())) + 1)
    y0 = min(y0, int(np.floor(points[:, 2].min())))
    y1 = max(y1, int(np.ceil(points[:, 2].max())) + 1)
    x0 = min(x0, int(np.floor(points[:, 3].min())))
    x1 = max(x1, int(np.ceil(points[:, 3].max())) + 1)

    t0, z0, y0, x0 = max(0, t0), max(0, z0), max(0, y0), max(0, x0)
    t1, z1, y1, x1 = min(image.shape[0], t1), min(image.shape[1], z1), min(image.shape[2], y1), min(image.shape[3], x1)

    crop = np.asarray(image[t0:t1, z0:z1, y0:y1, x0:x1])
    projection = crop.max(axis=1)
    origin = np.asarray([t0, z0, y0, x0], dtype=float)
    projected_points = np.asarray(
        [
            [candidate.parent.t - t0, candidate.parent.zyx[1] - y0, candidate.parent.zyx[2] - x0],
            [candidate.daughter_a.t - t0, candidate.daughter_a.zyx[1] - y0, candidate.daughter_a.zyx[2] - x0],
            [candidate.daughter_b.t - t0, candidate.daughter_b.zyx[1] - y0, candidate.daughter_b.zyx[2] - x0],
        ],
        dtype=float,
    )
    return projection, projected_points, origin


class DivisionLabeler:
    def __init__(self, viewer: napari.Viewer, image, candidates: list[DivisionCandidate], output_csv: Path, args: argparse.Namespace):
        self.viewer = viewer
        self.image = image
        self.candidates = candidates
        self.output_csv = output_csv
        self.args = args
        self.rows = read_existing_labels(output_csv)
        self.labels_by_key = self._existing_labels()
        self.index = 0
        self.status = QLabel("")

    def _existing_labels(self) -> dict[tuple[str, int, int, int], str]:
        labels = {}
        for row in self.rows:
            try:
                key = (
                    row["dataset"],
                    int(row["parent_node_id"]),
                    min(int(row["daughter_a_node_id"]), int(row["daughter_b_node_id"])),
                    max(int(row["daughter_a_node_id"]), int(row["daughter_b_node_id"])),
                )
                labels[key] = row["label"]
            except Exception:
                continue
        return labels

    def add_controls(self) -> None:
        widget = QWidget()
        layout = QVBoxLayout()
        help_label = QLabel("1 true division | 2 false | 3 unclear\nN next | P previous | S save")
        true_button = QPushButton("True division (1)")
        false_button = QPushButton("False / not division (2)")
        unclear_button = QPushButton("Unclear (3)")
        previous_button = QPushButton("Previous (P)")
        next_button = QPushButton("Next (N)")
        save_button = QPushButton("Save (S)")

        true_button.clicked.connect(lambda: self.set_label("true"))
        false_button.clicked.connect(lambda: self.set_label("false"))
        unclear_button.clicked.connect(lambda: self.set_label("unclear"))
        previous_button.clicked.connect(self.previous_candidate)
        next_button.clicked.connect(self.next_candidate)
        save_button.clicked.connect(self.save)

        for item in [help_label, true_button, false_button, unclear_button, previous_button, next_button, save_button, self.status]:
            layout.addWidget(item)
        widget.setLayout(layout)
        self.viewer.window.add_dock_widget(widget, area="right", name="Division labels")

    def bind_keys(self) -> None:
        @self.viewer.bind_key("1", overwrite=True)
        def _true(viewer):
            self.set_label("true")

        @self.viewer.bind_key("2", overwrite=True)
        def _false(viewer):
            self.set_label("false")

        @self.viewer.bind_key("3", overwrite=True)
        def _unclear(viewer):
            self.set_label("unclear")

        @self.viewer.bind_key("n", overwrite=True)
        def _next(viewer):
            self.next_candidate()

        @self.viewer.bind_key("p", overwrite=True)
        def _previous(viewer):
            self.previous_candidate()

        @self.viewer.bind_key("s", overwrite=True)
        def _save(viewer):
            self.save()

    def current_candidate(self) -> DivisionCandidate:
        return self.candidates[self.index]

    def show_candidate(self) -> None:
        self.viewer.layers.clear()
        candidate = self.current_candidate()
        projection, points, origin = crop_project_movie(self.image, candidate, self.args)
        parent_point = points[:1]
        daughter_points = points[1:]
        midpoint = np.asarray([[daughter_points[:, 0].mean(), daughter_points[:, 1].mean(), daughter_points[:, 2].mean()]])

        self.viewer.add_image(
            projection,
            name=f"{candidate.dataset} candidate {self.index}",
            scale=(1.0, 0.40625, 0.40625),
            contrast_limits=(self.args.contrast_min, self.args.contrast_max),
        )
        self.viewer.add_points(
            parent_point,
            ndim=3,
            name="parent at t",
            scale=(1.0, 0.40625, 0.40625),
            size=self.args.point_size,
            face_color="yellow",
            opacity=1.0,
        )
        self.viewer.add_points(
            daughter_points,
            ndim=3,
            name="daughter candidates at t+1",
            scale=(1.0, 0.40625, 0.40625),
            size=self.args.point_size,
            face_color="cyan",
            opacity=1.0,
        )
        self.viewer.add_points(
            midpoint,
            ndim=3,
            name="daughter midpoint",
            scale=(1.0, 0.40625, 0.40625),
            size=max(1.0, self.args.point_size * 0.6),
            face_color="magenta",
            opacity=0.85,
        )

        try:
            self.viewer.dims.axis_labels = ("frame", "crop_y", "crop_x")
            self.viewer.dims.current_step = (int(candidate.parent.t - origin[0]), 0, 0)
        except Exception:
            pass

        key = candidate_key(candidate)
        label = self.labels_by_key.get(key, "unlabeled")
        self.status.setText(
            f"{self.index + 1}/{len(self.candidates)} label={label}\n"
            f"source={candidate.source} score={candidate.score:.3f}\n"
            f"parent node={candidate.parent.node_id} t={candidate.parent.t} zyx={candidate.parent.zyx.round(1).tolist()}\n"
            f"A node={candidate.daughter_a.node_id} t={candidate.daughter_a.t} zyx={candidate.daughter_a.zyx.round(1).tolist()}\n"
            f"B node={candidate.daughter_b.node_id} t={candidate.daughter_b.t} zyx={candidate.daughter_b.zyx.round(1).tolist()}"
        )

    def set_label(self, label: str) -> None:
        candidate = self.current_candidate()
        key = candidate_key(candidate)
        self.labels_by_key[key] = label
        new_row = make_label_row(candidate, label)
        replaced = False
        for i, row in enumerate(self.rows):
            try:
                row_key = (
                    row["dataset"],
                    int(row["parent_node_id"]),
                    min(int(row["daughter_a_node_id"]), int(row["daughter_b_node_id"])),
                    max(int(row["daughter_a_node_id"]), int(row["daughter_b_node_id"])),
                )
            except Exception:
                continue
            if row_key == key:
                self.rows[i] = new_row
                replaced = True
                break
        if not replaced:
            self.rows.append(new_row)
        self.save()
        self.next_candidate()

    def next_candidate(self) -> None:
        self.index = min(self.index + 1, len(self.candidates) - 1)
        self.show_candidate()

    def previous_candidate(self) -> None:
        self.index = max(self.index - 1, 0)
        self.show_candidate()

    def save(self) -> None:
        write_labels(self.output_csv, self.rows)
        print(f"saved {len(self.rows)} labels to {self.output_csv}")


def main() -> None:
    args = parse_args()
    image = zarr.open(args.zarr_path / args.array_path, mode="r")
    dataset = args.zarr_path.stem

    candidates: list[DivisionCandidate] = []
    if args.geff is not None:
        geff_candidates = load_geff_divisions(args.geff, dataset)
        candidates.extend(geff_candidates)
        print(f"loaded {len(geff_candidates)} GEFF true division candidates")
    if args.result_dir is not None:
        mined = mine_candidates(args.result_dir, dataset, args.max_candidates, args)
        candidates.extend(mined)
        print(f"mined {len(mined)} candidate divisions from {args.result_dir}")
    if not candidates:
        raise ValueError("No candidates found. Provide --result-dir and/or --geff.")

    deduped = []
    seen = set()
    for candidate in candidates:
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    candidates = deduped[: args.max_candidates]

    print(f"labeling {len(candidates)} candidates")
    viewer = napari.Viewer()
    labeler = DivisionLabeler(viewer, image, candidates, args.output_csv, args)
    labeler.add_controls()
    labeler.bind_keys()
    labeler.show_candidate()
    napari.run()


if __name__ == "__main__":
    main()
