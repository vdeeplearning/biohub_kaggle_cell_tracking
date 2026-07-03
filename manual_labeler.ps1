param(
    [string]$TrainDir = ".\data\train",

    [string]$OutputCsv = ".\manual_labels\manual_centroids.csv",

    [string]$Sample = "",

    [double]$ContrastMin = 0,
    [double]$ContrastMax = 3500,

    [int]$Seed = -1
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\manual_point_labeler.py",
    "--train-dir", $TrainDir,
    "--output-csv", $OutputCsv,
    "--contrast-min", "$ContrastMin",
    "--contrast-max", "$ContrastMax"
)

if ($Sample -ne "") {
    $ArgsList += @("--sample", $Sample)
}

if ($Seed -ge 0) {
    $ArgsList += @("--seed", "$Seed")
}

& $Python @ArgsList
