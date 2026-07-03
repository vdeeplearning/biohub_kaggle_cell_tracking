param(
    [Parameter(Mandatory = $true)]
    [string]$ZarrPath,

    [Parameter(Mandatory = $true)]
    [string]$GeffPath,

    [int]$CropSize = 96,

    [double]$ContrastMin = [double]::NaN,
    [double]$ContrastMax = [double]::NaN
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\view_geff_label_movie.py",
    $ZarrPath,
    $GeffPath,
    "--crop-size",
    "$CropSize"
)

if (-not [double]::IsNaN($ContrastMin)) {
    $ArgsList += @("--contrast-min", "$ContrastMin")
}

if (-not [double]::IsNaN($ContrastMax)) {
    $ArgsList += @("--contrast-max", "$ContrastMax")
}

& $Python @ArgsList
