$ErrorActionPreference = "Stop"

$BasePython = & ".\scripts\resolve_python.ps1"

if (-not (Test-Path -LiteralPath ".\.venv_viewer\Scripts\python.exe")) {
    & $BasePython -m venv .venv_viewer
}

& ".\.venv_viewer\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv_viewer\Scripts\python.exe" -m pip install -r requirements-viewer.txt
