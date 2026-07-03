from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("biohub_classical_submission.ipynb")
TARGET = Path("biohub_classical_submission_fast.ipynb")


FAST_DISCOVERY_CELL = [
    "def find_test_zarrs(input_root: Path) -> list[Path]:\n",
    "    known = input_root / 'competitions' / 'biohub-cell-tracking-during-development' / 'test'\n",
    "    if known.exists():\n",
    "        return sorted(known.glob('*.zarr'))\n",
    "\n",
    "    # Fallback for unexpected Kaggle input nesting. This is slower, so avoid it when possible.\n",
    "    for test_dir in input_root.glob('**/test'):\n",
    "        zarrs = sorted(test_dir.glob('*.zarr'))\n",
    "        if zarrs:\n",
    "            return zarrs\n",
    "    return sorted(input_root.rglob('*.zarr'))\n",
    "\n",
    "\n",
    "test_zarrs = find_test_zarrs(INPUT_ROOT)\n",
    "print(f'found {len(test_zarrs)} zarr test samples')\n",
    "for path in test_zarrs[:10]:\n",
    "    print(path)\n",
    "if len(test_zarrs) > 10:\n",
    "    print(f'... {len(test_zarrs) - 10} more samples')\n",
    "if not test_zarrs:\n",
    "    raise FileNotFoundError('No .zarr samples found under /kaggle/input')\n",
]


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("def find_test_zarrs("):
            cell["source"] = FAST_DISCOVERY_CELL
        if "def build_nodes(" in source:
            cell["source"] = [
                line.replace(
                    "        print(f'{zarr_path.stem} t={t:03d}: {len(peaks)} peaks')\n",
                    "        if t % 10 == 0 or t == int(image.shape[0]) - 1:\n"
                    "            print(f'{zarr_path.stem} t={t:03d}: {len(peaks)} peaks')\n",
                )
                for line in cell["source"]
            ]

    TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
