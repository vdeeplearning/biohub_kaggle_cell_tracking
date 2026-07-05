param(
    [Parameter(Mandatory = $true)]
    [string]$ZarrPath,

    [Parameter(Mandatory = $true)]
    [string]$GeffPath,

    [int]$DivisionIndex = 0,

    [switch]$OnlyDivisionIndex,

    [int]$TimeRadius = 3,

    [string]$CropRadiusZyx = "8,48,48",

    [ValidateSet("auto", "xy", "xz", "yz", "volume")]
    [string]$Plane = "auto",

    [double]$ContrastMin = [double]::NaN,
    [double]$ContrastMax = [double]::NaN,

    [double]$PointSize = [double]::NaN
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\view_geff_divisions.py",
    $ZarrPath,
    $GeffPath,
    "--division-index",
    "$DivisionIndex",
    "--time-radius",
    "$TimeRadius",
    "--crop-radius-zyx",
    "$CropRadiusZyx",
    "--plane",
    "$Plane"
)

if ($OnlyDivisionIndex) {
    $ArgsList += "--only-division-index"
}

if (-not [double]::IsNaN($ContrastMin)) {
    $ArgsList += @("--contrast-min", "$ContrastMin")
}

if (-not [double]::IsNaN($ContrastMax)) {
    $ArgsList += @("--contrast-max", "$ContrastMax")
}

if (-not [double]::IsNaN($PointSize)) {
    $ArgsList += @("--point-size", "$PointSize")
}

& $Python @ArgsList
