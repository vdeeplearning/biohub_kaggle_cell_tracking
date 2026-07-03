from __future__ import annotations

import argparse
import csv
from pathlib import Path


SUBMISSION_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert classical tracking outputs into Kaggle submission CSV.")
    parser.add_argument(
        "result_dirs",
        type=Path,
        nargs="+",
        help="One or more output directories containing nodes.csv and edges.csv.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for result_dir in args.result_dirs:
        nodes_path = result_dir / "nodes.csv"
        edges_path = result_dir / "edges.csv"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Missing {nodes_path}")
        if not edges_path.exists():
            raise FileNotFoundError(f"Missing {edges_path}")

        nodes = read_csv(nodes_path)
        edges = read_csv(edges_path)

        for node in nodes:
            rows.append(
                {
                    "dataset": node["dataset"],
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": int(node["z"]),
                    "y": int(node["y"]),
                    "x": int(node["x"]),
                    "source_id": -1,
                    "target_id": -1,
                }
            )

        for edge in edges:
            rows.append(
                {
                    "dataset": edge["dataset"],
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": int(edge["source_id"]),
                    "target_id": int(edge["target_id"]),
                }
            )

    for row_id, row in enumerate(rows):
        row["id"] = row_id

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    node_count = sum(1 for row in rows if row["row_type"] == "node")
    edge_count = sum(1 for row in rows if row["row_type"] == "edge")
    datasets = sorted({str(row["dataset"]) for row in rows})
    print(f"wrote {args.out}")
    print(f"datasets={len(datasets)}, nodes={node_count}, edges={edge_count}, rows={len(rows)}")
    for dataset in datasets:
        print(f"  {dataset}")


if __name__ == "__main__":
    main()
