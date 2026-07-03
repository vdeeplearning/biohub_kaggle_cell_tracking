param(
    [double]$LinkMaxDistanceUm = 7.0,
    [switch]$EnableGlobalFlow,
    [double]$FlowConfidentDistanceUm = 4.0
)

$ErrorActionPreference = "Stop"
$Python = & ".\scripts\resolve_python.ps1"

powershell -ExecutionPolicy Bypass -File .\run_classical_one.ps1 `
    -ZarrPath ".\data\train\44b6_0113de3b.zarr" `
    -GeffPath ".\data\train\44b6_0113de3b.geff" `
    -OutDir ".\outputs\classical_44b6_0113de3b_q960" `
    -MaxFrames 100 `
    -ThresholdQuantile 96.0 `
    -MinPeakDistance 3 `
    -MaxPeaksPerFrame 3000 `
    -LinkMaxDistanceUm $LinkMaxDistanceUm `
    -EnableGlobalFlow:$EnableGlobalFlow `
    -FlowConfidentDistanceUm $FlowConfidentDistanceUm

powershell -ExecutionPolicy Bypass -File .\run_classical_one.ps1 `
    -ZarrPath ".\data\train\44b6_0b24845f.zarr" `
    -GeffPath ".\data\train\44b6_0b24845f.geff" `
    -OutDir ".\outputs\classical_44b6_0b24845f_q660" `
    -MaxFrames 100 `
    -ThresholdQuantile 66.0 `
    -MinPeakDistance 3 `
    -MaxPeaksPerFrame 3000 `
    -LinkMaxDistanceUm $LinkMaxDistanceUm `
    -EnableGlobalFlow:$EnableGlobalFlow `
    -FlowConfidentDistanceUm $FlowConfidentDistanceUm

& $Python .\scripts\make_submission.py `
    .\outputs\classical_44b6_0113de3b_q960 `
    .\outputs\classical_44b6_0b24845f_q660 `
    --out .\outputs\submission_two_train_samples_classical.csv
