param(
    [string]$CampaignName = "native_serial",
    [int]$GrindingTrials = 256,
    [int]$GrindingWorkers = 4,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$ResultRoot = Join-Path $CodeRoot "results"
$ReviewRoot = Join-Path $CodeRoot "validation\outputs\campaigns\$CampaignName"

New-Item -ItemType Directory -Force -Path $ResultRoot, $ReviewRoot | Out-Null
$env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "overview\runs\serial"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:GOMAXPROCS = "4"
$env:COMMITJL_CHECKPOINT_ROWS = "1"

function Invoke-CampaignStep {
    param([string]$Name, [string[]]$Arguments)
    $LogPath = Join-Path $env:COMMITJL_RESULTS_DIR "logs\$Name.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    Write-Host "[$(Get-Date -Format s)] Starting $Name"
    # Windows PowerShell 5 wraps any native stderr output as a nonterminating
    # NativeCommandError. tqdm writes its normal progress display to stderr,
    # so temporarily allow that stream and decide success from the process
    # exit code instead.
    $PreviousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $ProcessExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorAction
    }
    if ($ProcessExitCode -ne 0) {
        throw "$Name failed with exit code $ProcessExitCode. See $LogPath. Rerun the same command after fixing the error; completed rows are checkpointed."
    }
}

Push-Location $CodeRoot
try {
    if (-not $SkipChecks) {
        Invoke-CampaignStep "python_preflight" @("-m", "pytest", "tests", "--ignore=tests/test_zk.py", "-q")
        Push-Location (Join-Path $CodeRoot "zk")
        try {
            & go build ./...
            if ($LASTEXITCODE -ne 0) { throw "go build failed" }
            & go test -count=1 ./...
            if ($LASTEXITCODE -ne 0) { throw "go tests failed" }
        }
        finally { Pop-Location }
    }

    $env:COMMITJL_CHECKPOINT_ROWS = "8"

    # RQ1: fidelity across datasets, projections, m, q, k, and ten seeds.
    $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq1\runs\native"
    Invoke-CampaignStep "rq1_fidelity" @("-m", "experiments.rq1_fidelity", "--config", "experiments/configs/paper_rq1.yaml")

    # RQ3: protocol ordering/randomness overhead and grinding sensitivity.
    $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq3\runs\serial"
    Invoke-CampaignStep "rq3_overhead" @("-m", "experiments.rq3_overhead", "--config", "experiments/configs/paper_rq3.yaml")
    Invoke-CampaignStep "rq3_grinding" @(
        "-m", "experiments.rq3_grinding", "--config", "experiments/configs/paper_rq3_grinding.yaml",
        "--n-trials", "$GrindingTrials", "--Q-list", "1,2,4,8,16,32", "--workers", "$GrindingWorkers"
    )

    # RQ4: centroid/DBSCAN certificates plus the registered phase diagram.
    $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq4\runs\serial"
    Invoke-CampaignStep "rq4_certificates" @(
        "-m", "experiments.rq4_certificates", "--config", "experiments/configs/paper_rq4.yaml", "--phase-diagram"
    )

    # RQ5: controlled one-factor-at-a-time native scaling.
    $env:COMMITJL_RESULTS_DIR = Join-Path $ResultRoot "rq5\runs\serial"
    Invoke-CampaignStep "rq5_scaling" @("-m", "experiments.rq5_scaling", "--config", "experiments/configs/paper_rq5.yaml")

    $Manifest = [ordered]@{
        campaign = $CampaignName
        completed_at = (Get-Date).ToString("o")
        results = $ResultRoot
        included = @("RQ1", "RQ3-overhead", "RQ3-grinding", "RQ4", "RQ5")
        excluded = @("RQ2 proof cost", "proof-based ablation")
        grinding_trials = $GrindingTrials
        grinding_workers = $GrindingWorkers
        checkpoint = "atomic parquet every 8 rows and on normal process exit; rerun with the same CampaignName to resume"
    }
    $Manifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $ReviewRoot "campaign_manifest.json")
    Write-Host "Campaign completed. Results: $ResultRoot"
}
finally { Pop-Location }
