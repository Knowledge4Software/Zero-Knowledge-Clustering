param(
    [ValidateRange(1,6)][int]$Workers = 3,
    [switch]$Smoke
)
$ErrorActionPreference = 'Stop'
$CodeRoot = Split-Path -Parent $PSScriptRoot
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:GOMAXPROCS = '4'
$env:PYTHONUNBUFFERED = '1'
$Campaign = if ($Smoke) { 'smoke_v1' } else { 'local_v1' }
$Out = Join-Path $CodeRoot "results\diagnostics\$Campaign"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Arguments = @('-m','campaigns.diagnostic','--workers',"$Workers",'--out',$Out)
if ($Smoke) { $Arguments += '--smoke' }
Push-Location $CodeRoot
try {
    $ErrorActionPreference = 'Continue'
    & python @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $Out 'campaign.log') -Append
    $RunExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($RunExitCode -ne 0) { throw "Diagnostic campaign failed (exit $RunExitCode). Checkpoints retained under $Out." }
    Write-Host "Report: $Out\analysis\interpretation.md"
} finally { Pop-Location }
