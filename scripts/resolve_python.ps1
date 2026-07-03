$Candidates = @(
    ".\.venv_viewer\Scripts\python.exe",
    ".\.venv\Scripts\python.exe",
    "python",
    "py",
    "python3",
    "C:\Users\tommy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

foreach ($Candidate in $Candidates) {
    if ($Candidate.EndsWith(".exe")) {
        if (Test-Path -LiteralPath $Candidate) {
            try {
                & $Candidate --version *> $null
                if ($LASTEXITCODE -eq 0) {
                    return $Candidate
                }
            }
            catch {
            }
        }
    }
    else {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
        if ($null -ne $Command) {
            try {
                & $Candidate --version *> $null
                if ($LASTEXITCODE -eq 0) {
                    return $Candidate
                }
            }
            catch {
            }
        }
    }
}

throw "No Python executable found. Install Python or run this from Codex where bundled Python is available."
