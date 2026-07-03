from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("biohub_classical_submission.ipynb")


IMPORT_CELL = [
    "from __future__ import annotations\n",
    "\n",
    "import csv\n",
    "import json\n",
    "from collections import defaultdict\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from scipy.ndimage import gaussian_filter\n",
    "from scipy.optimize import linear_sum_assignment\n",
    "from skimage.feature import peak_local_max\n",
    "\n",
    "try:\n",
    "    from numcodecs import Blosc\n",
    "\n",
    "    def decode_blosc(data: bytes) -> bytes:\n",
    "        return Blosc().decode(data)\n",
    "\n",
    "    BLOSC_BACKEND = 'numcodecs'\n",
    "except Exception:\n",
    "    try:\n",
    "        import blosc2\n",
    "\n",
    "        def decode_blosc(data: bytes) -> bytes:\n",
    "            return blosc2.decompress(data)\n",
    "\n",
    "        BLOSC_BACKEND = 'blosc2'\n",
    "    except Exception as exc:\n",
    "        raise ModuleNotFoundError(\n",
    "            'Need numcodecs or blosc2 to decode Biohub Zarr chunks in Kaggle no-internet mode.'\n",
    "        ) from exc\n",
    "\n",
    "print('numpy', np.__version__)\n",
    "print('pandas', pd.__version__)\n",
    "print('blosc backend', BLOSC_BACKEND)\n",
]


ZARR_READER_CELL = [
    "class SimpleZarr3Array:\n",
    "    def __init__(self, array_path: Path):\n",
    "        self.array_path = Path(array_path)\n",
    "        with (self.array_path / 'zarr.json').open(encoding='utf-8') as f:\n",
    "            self.meta = json.load(f)\n",
    "\n",
    "        self.shape = tuple(int(v) for v in self.meta['shape'])\n",
    "        chunk_grid = self.meta['chunk_grid']['configuration']\n",
    "        self.chunk_shape = tuple(int(v) for v in chunk_grid['chunk_shape'])\n",
    "        self.dtype = self._dtype_from_meta(self.meta.get('data_type') or self.meta.get('dtype'))\n",
    "\n",
    "    @staticmethod\n",
    "    def _dtype_from_meta(data_type) -> np.dtype:\n",
    "        text = str(data_type).lower()\n",
    "        mapping = {\n",
    "            'uint8': 'u1', 'uint16': '<u2', 'uint32': '<u4', 'uint64': '<u8',\n",
    "            'int8': 'i1', 'int16': '<i2', 'int32': '<i4', 'int64': '<i8',\n",
    "            'float32': '<f4', 'float64': '<f8',\n",
    "        }\n",
    "        if text not in mapping:\n",
    "            raise ValueError(f'Unsupported Zarr dtype: {data_type}')\n",
    "        return np.dtype(mapping[text])\n",
    "\n",
    "    def _read_chunk(self, chunk_indices: tuple[int, int, int, int]) -> np.ndarray:\n",
    "        chunk_path = (\n",
    "            self.array_path / 'c' / str(chunk_indices[0]) / str(chunk_indices[1]) /\n",
    "            str(chunk_indices[2]) / str(chunk_indices[3])\n",
    "        )\n",
    "        if not chunk_path.exists():\n",
    "            return np.zeros(self.chunk_shape, dtype=self.dtype)\n",
    "        decoded = decode_blosc(chunk_path.read_bytes())\n",
    "        return np.frombuffer(decoded, dtype=self.dtype).reshape(self.chunk_shape)\n",
    "\n",
    "    def __getitem__(self, item):\n",
    "        # This notebook only needs whole timepoint reads: image[t, :, :, :].\n",
    "        if not isinstance(item, tuple) or len(item) != 4 or not isinstance(item[0], (int, np.integer)):\n",
    "            raise NotImplementedError('SimpleZarr3Array supports image[t, :, :, :] reads only')\n",
    "        t = int(item[0])\n",
    "        z_slice, y_slice, x_slice = item[1], item[2], item[3]\n",
    "        if z_slice != slice(None) or y_slice != slice(None) or x_slice != slice(None):\n",
    "            raise NotImplementedError('SimpleZarr3Array supports full Z/Y/X timepoint reads only')\n",
    "        chunk = self._read_chunk((t, 0, 0, 0))\n",
    "        return chunk[0, : self.shape[1], : self.shape[2], : self.shape[3]]\n",
    "\n",
    "\n",
    "def open_zarr_array(zarr_path: Path, array_path: str = '0') -> SimpleZarr3Array:\n",
    "    return SimpleZarr3Array(Path(zarr_path) / array_path)\n",
]


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[1]["source"] = IMPORT_CELL

    cells[:] = [
        cell
        for cell in cells
        if "class SimpleZarr3Array" not in "".join(cell.get("source", []))
    ]
    cells.insert(4, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ZARR_READER_CELL})

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        cell["source"] = [
            line.replace("image = zarr.open(zarr_path / array_path, mode='r')", "image = open_zarr_array(zarr_path, array_path)")
            for line in cell["source"]
        ]

    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
