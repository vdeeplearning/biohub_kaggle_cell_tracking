param(
    [Parameter(Mandatory = $true)]
    [string]$ZarrPath,

    [string]$GeffPath = "",

    [double]$ContrastMin = [double]::NaN,
    [double]$ContrastMax = [double]::NaN,

    [double]$PointSize = [double]::NaN
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @("scripts\view_zarr.py", $ZarrPath)

if ($GeffPath -ne "") {
    $ArgsList += @("--geff", $GeffPath)
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
