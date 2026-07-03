# Biohub Cell Tracking Workspace

Small local workspace for exploring the Biohub cell tracking competition data.

## View A Zarr Volume

Install the viewer dependencies once:

```powershell
.\setup_viewer.ps1
```

Open a copied sample `.zarr`:

```powershell
.\view_zarr.ps1 "C:\path\to\sample.zarr"
```

Open it with matching sparse ground-truth centroids overlaid:

```powershell
.\view_zarr.ps1 "C:\path\to\sample.zarr" -GeffPath "C:\path\to\sample.geff"
```

The label overlay is sparse. The viewer prints the first labeled centroid location
and tries to jump to it. You can enlarge the point overlay:

```powershell
.\view_zarr.ps1 "C:\path\to\sample.zarr" -GeffPath "C:\path\to\sample.geff" -PointSize 16
```

If contrast looks washed out, pass limits:

```powershell
.\view_zarr.ps1 "C:\path\to\sample.zarr" -ContrastMin 0 -ContrastMax 2000
```

The viewer uses napari. The image axes are `(T, Z, Y, X)`, with physical scale
`z=1.625`, `y=0.40625`, `x=0.40625` microns per voxel.

## Follow Sparse Labels

To avoid hunting through the full 4D volume, open a movie of 2D crops centered
on each sparse GEFF label:

```powershell
.\view_label_movie.ps1 ".\data\train\44b6_0113de3b.zarr" ".\data\train\44b6_0113de3b.geff" -CropSize 96 -ContrastMin 0 -ContrastMax 2000
```

The first axis is `label_frame`, not the original timepoint. The script prints
the mapping from `label_frame` to `node_id,t,z,y,x`.

Use `-CropSize 0` to show the full 256x256 slice at each label instead of a
centered crop:

```powershell
.\view_label_movie.ps1 ".\data\train\44b6_0113de3b.zarr" ".\data\train\44b6_0113de3b.geff" -CropSize 0 -ContrastMin 0 -ContrastMax 2000
```

## Classical Baseline

Run a classical 3D peak detector and adjacent-frame linker on one local sample:

```powershell
.\run_classical_one.ps1
```

This writes:

```text
outputs/classical_44b6_0113de3b/
  nodes.csv
  edges.csv
  metrics.json
```

Open a 4D overlay with predicted detections, track colors, node numbers, track
tails, and sparse GEFF labels:

```powershell
.\view_classical_overlay.ps1
```

Add `-ShowNodeIds` if you want node numbers drawn on top of the detections.

The sparse-label metrics are diagnostic only. Many real cells are unlabeled, so
the labeled-subset precision values are not true full-dataset precision.

Enable the conservative classical division pass:

```powershell
.\run_classical_one.ps1 -EnableDivisions
```

Accepted one-parent/two-daughter hypotheses are written to `divisions.csv`.
By default this only branches parents with no accepted normal continuation. Add
`-DivisionAllowParentRewrite` to let a division replace a one-child continuation.
The default division rules are intentionally strict: at most two nearby daughter
candidates, persistent daughter tracks, and a daughter midpoint close to parent.
