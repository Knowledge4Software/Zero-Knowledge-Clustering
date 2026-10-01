param(
    [ValidateSet("RQ3Overhead", "RQ3Grinding", "RQ4", "RQ5", "All")]
    [string]$Stage = "All",
    [ValidateRange(1, 16)]
    [int]$Workers = 6
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$SourceRoot = Join-Path $CodeRoot "results\rq1\runs\native"
$ResultRoot = Join-Path $CodeRoot "results"

$oldCampaign = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*experiments.rq3_overhead*' -and $_.Name -like 'python*'
}
if ($oldCampaign) {
    throw "The serial RQ3 campaign is still running. Stop it in its terminal with Ctrl+C before starting this parallel campaign."
}

# Keep the completed RQ1 rows in their own run directory. Serial and parallel
# latency observations must not share one RQ3 table.
if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot "rq1_fidelity.parquet"))) {
    throw "Completed RQ1 file missing under $SourceRoot"
}
$metadataPath = Join-Path $ResultRoot "overview\runs\native_parallel_metadata.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $metadataPath) | Out-Null
if (Test-Path -LiteralPath $metadataPath) {
    $existingMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ([int]$existingMetadata.rq3_parallel_workers -ne $Workers) {
        throw "This campaign was started with $($existingMetadata.rq3_parallel_workers) workers. Resume with the same -Workers value."
    }
}
$metadata = [ordered]@{
    campaign = "paper_native_parallel_v1"
    rq1_source = "rq1/runs/native"
    rq3_execution_mode = "parallel_processes"
    rq3_parallel_workers = $Workers
    rq3_timing_semantics = "per-job elapsed time under CPU contention; source comparisons use the same worker count"
    rq3_beacon = "Simulated local trace"
}
$metadata | ConvertTo-Json -Depth 3 | Set-Content -Encoding UTF8 $metadataPath
$env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq3\runs\native"
$env:COMMITJL_CHECKPOINT_ROWS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:GOMAXPROCS = "4"

function Invoke-CampaignStep {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $env:COMMITJL_RESULTS_DIR "logs\$Name.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
    Write-Host "[$(Get-Date -Format s)] Starting $Name"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @Arguments 2>&1 | Tee-Object -FilePath $log
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($exitCode -ne 0) { throw "$Name failed (exit $exitCode); see $log" }
}

Push-Location $CodeRoot
try {
    if ($Stage -in @("RQ3Overhead", "All")) {
        $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq3\runs\native"
        Invoke-CampaignStep "rq3_overhead_parallel" @(
            "-m", "campaigns.rq3_parallel", "--config", "experiments/configs/paper_rq3.yaml",
            "--out", (Join-Path $env:COMMITJL_RESULTS_DIR "rq3_overhead.parquet"), "--workers", "$Workers"
        )
    }
    if ($Stage -in @("RQ3Grinding", "All")) {
        $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq3\runs\native"
        Invoke-CampaignStep "rq3_grinding" @(
            "-m", "experiments.rq3_grinding", "--config", "experiments/configs/paper_rq3_grinding.yaml",
            "--n-trials", "256", "--Q-list", "1,2,4,8,16,32", "--workers", "$Workers"
        )
    }
    if ($Stage -in @("RQ4", "All")) {
        $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq4\runs\native"
        Invoke-CampaignStep "rq4_certificates" @(
            "-m", "experiments.rq4_certificates", "--config", "experiments/configs/paper_rq4.yaml", "--phase-diagram"
        )
    }
    if ($Stage -in @("RQ5", "All")) {
        $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq5\runs\native"
        Invoke-CampaignStep "rq5_scaling" @(
            "-m", "experiments.rq5_scaling", "--config", "experiments/configs/paper_rq5.yaml"
        )
    }
    Write-Host "[$(Get-Date -Format s)] $Stage complete: $ResultRoot"
}
finally { Pop-Location }
