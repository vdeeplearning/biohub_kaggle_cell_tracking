param(
    [string]$ZarrPath = ".\data\train\44b6_0113de3b.zarr",
    [string]$ResultDir = ".\outputs\classical_44b6_0113de3b",
    [string]$GeffPath = ".\data\train\44b6_0113de3b.geff",
    [double]$ContrastMin = 0,
    [double]$ContrastMax = 2000,
    [double]$PointSize = 7,
    [switch]$ShowNodeIds
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\view_classical_overlay.py",
    $ZarrPath,
    $ResultDir,
    "--geff",
    $GeffPath,
    "--contrast-min",
    "$ContrastMin",
    "--contrast-max",
    "$ContrastMax",
    "--point-size",
    "$PointSize"
)

if ($ShowNodeIds) {
    $ArgsList += @("--show-node-ids")
}

& $Python @ArgsList
