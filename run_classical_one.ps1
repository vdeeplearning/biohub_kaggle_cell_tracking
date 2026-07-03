param(
    [string]$ZarrPath = ".\data\train\44b6_0113de3b.zarr",
    [string]$GeffPath = ".\data\train\44b6_0113de3b.geff",
    [string]$OutDir = ".\outputs\classical_44b6_0113de3b",
    [int]$MaxFrames = 100,
    [switch]$Profile,
    [double]$ThresholdQuantile = 99.92,
    [int]$MinPeakDistance = 4,
    [int]$MaxPeaksPerFrame = 1500,
    [switch]$EnableBlobSizeFilter,
    [string]$BlobSizeMode = "hard",
    [int]$BlobMinVoxels = 450,
    [int]$BlobMaxVoxels = 1400,
    [double]$BlobTargetVoxels = 928.0,
    [double]$BlobSizeSigmaVoxels = 350.0,
    [double]$BlobSizePenaltyWeight = 1.0,
    [double]$BlobFilterOversampleFactor = 2.0,
    [double]$BlobAlpha = 0.35,
    [double]$BlobBackgroundPercentile = 20.0,
    [string]$BlobCropRadiusZyx = "4,8,8",
    [double]$LinkMaxDistanceUm = 7.0,
    [switch]$EnableGlobalFlow,
    [double]$FlowConfidentDistanceUm = 4.0,
    [switch]$EnableDivisions,
    [double]$DivisionMaxParentDistanceUm = 10.0,
    [double]$DivisionMinDaughterSeparationUm = 2.0,
    [double]$DivisionMaxDaughterSeparationUm = 15.0,
    [double]$DivisionMaxMidpointDistanceUm = 4.0,
    [int]$DivisionMaxNearbyCandidates = 2,
    [int]$DivisionPersistenceFrames = 3,
    [switch]$DivisionAllowParentRewrite
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

$ArgsList = @(
    "scripts\classical_detect_track.py",
    $ZarrPath,
    "--geff",
    $GeffPath,
    "--out-dir",
    $OutDir,
    "--max-frames",
    "$MaxFrames",
    "--threshold-quantile",
    "$ThresholdQuantile",
    "--min-peak-distance",
    "$MinPeakDistance",
    "--max-peaks-per-frame",
    "$MaxPeaksPerFrame",
    "--link-max-distance-um",
    "$LinkMaxDistanceUm"
)

if ($Profile) {
    $ArgsList += @("--profile")
}

if ($EnableBlobSizeFilter) {
    $ArgsList += @(
        "--enable-blob-size-filter",
        "--blob-size-mode",
        "$BlobSizeMode",
        "--blob-min-voxels",
        "$BlobMinVoxels",
        "--blob-max-voxels",
        "$BlobMaxVoxels",
        "--blob-target-voxels",
        "$BlobTargetVoxels",
        "--blob-size-sigma-voxels",
        "$BlobSizeSigmaVoxels",
        "--blob-size-penalty-weight",
        "$BlobSizePenaltyWeight",
        "--blob-filter-oversample-factor",
        "$BlobFilterOversampleFactor",
        "--blob-alpha",
        "$BlobAlpha",
        "--blob-background-percentile",
        "$BlobBackgroundPercentile",
        "--blob-crop-radius-zyx",
        "$BlobCropRadiusZyx"
    )
}

if ($EnableGlobalFlow) {
    $ArgsList += @(
        "--enable-global-flow",
        "--flow-confident-distance-um",
        "$FlowConfidentDistanceUm"
    )
}

if ($EnableDivisions) {
    $ArgsList += @(
        "--enable-divisions",
        "--division-max-parent-distance-um",
        "$DivisionMaxParentDistanceUm",
        "--division-min-daughter-separation-um",
        "$DivisionMinDaughterSeparationUm",
        "--division-max-daughter-separation-um",
        "$DivisionMaxDaughterSeparationUm",
        "--division-max-midpoint-distance-um",
        "$DivisionMaxMidpointDistanceUm",
        "--division-max-nearby-candidates",
        "$DivisionMaxNearbyCandidates",
        "--division-persistence-frames",
        "$DivisionPersistenceFrames"
    )
    if ($DivisionAllowParentRewrite) {
        $ArgsList += @("--division-allow-parent-rewrite")
    }
}

& $Python @ArgsList
