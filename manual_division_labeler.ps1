param(
    [Parameter(Mandatory = $true)]
    [string]$ZarrPath,

    [string]$ResultDir = "",

    [string]$GeffPath = "",

    [string]$OutputCsv = ".\manual_labels\manual_divisions.csv",

    [int]$MaxCandidates = 300,

    [int]$TimeBefore = 2,
    [int]$TimeAfter = 4,

    [int]$ZRadius = 4,
    [int]$XyRadius = 48,

    [double]$ContrastMin = 0,
    [double]$ContrastMax = 3500,

    [double]$PointSize = 5
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\manual_division_labeler.py",
    $ZarrPath,
    "--output-csv", $OutputCsv,
    "--max-candidates", "$MaxCandidates",
    "--time-before", "$TimeBefore",
    "--time-after", "$TimeAfter",
    "--z-radius", "$ZRadius",
    "--xy-radius", "$XyRadius",
    "--contrast-min", "$ContrastMin",
    "--contrast-max", "$ContrastMax",
    "--point-size", "$PointSize"
)

if ($ResultDir -ne "") {
    $ArgsList += @("--result-dir", $ResultDir)
}

if ($GeffPath -ne "") {
    $ArgsList += @("--geff", $GeffPath)
}

& $Python @ArgsList
