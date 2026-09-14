# Run from any directory. Python should point to your prepared environment.
param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath "data/synthetic")) {
        & $Python scripts/synthetic_demo.py
        if ($LASTEXITCODE -ne 0) { throw "Synthetic data generation failed" }
    }
    $runName = "outputs/demo-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    & $Python -m dual_fisheye run --config configs/synthetic.json --mode debug --output $runName
    if ($LASTEXITCODE -ne 0) { throw "Calibration pipeline failed" }
    Write-Output "Results: $PSScriptRoot/$runName"
} finally {
    Pop-Location
}
