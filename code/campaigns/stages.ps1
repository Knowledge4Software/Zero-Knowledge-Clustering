param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("AccuracyPilot", "PrecisionSweep", "FullNative", "RepresentativeProof", "ProofAblation", "FinalReport")]
    [string]$Stage,
    [int]$GrindingWorkers = 8,
    [int]$MaxProofShapes = 2,
    [ValidateSet("Parallel", "Serial")]
    [string]$NativeMode = "Parallel",
    [string]$NativeCampaignName = ""
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$ResultsRoot = Join-Path $CodeRoot "results"
if ($NativeCampaignName) {
    # Accept the campaign names used by earlier saved commands.
    $NativeMode = switch ($NativeCampaignName) {
        "paper_native_parallel_v1" { "Parallel" }
        "paper_native_rqs_v2" { "Serial" }
        default { throw "Unknown legacy campaign '$NativeCampaignName'; use -NativeMode Parallel or Serial" }
    }
}
$PilotRoot = Join-Path $ResultsRoot "rq1\runs\pilot"
$PrecisionRoot = Join-Path $ResultsRoot "rq5\runs\precision_20ref"
$RQ1Root = Join-Path $ResultsRoot "rq1\runs\native"
$NativeKind = if ($NativeMode -eq "Parallel") { "native" } else { "serial" }
$RQ3Root = Join-Path $ResultsRoot "rq3\runs\$NativeKind"
$RQ4Root = Join-Path $ResultsRoot "rq4\runs\$NativeKind"
$RQ5Root = Join-Path $ResultsRoot "rq5\runs\$NativeKind"
$ProofRoot = Join-Path $ResultsRoot "rq2\runs\representative"
$AblationRoot = Join-Path $ResultsRoot "rq2\runs\ablation"
$ReportRoot = Join-Path $ResultsRoot "overview\runs\final_report"
$SelectedConfig = Join-Path $ProofRoot "selected_rq1_shapes.yaml"

# One native experiment process at a time; threaded grinding uses its own
# worker count. Go receives a bounded scheduler budget for laptop RAM/CPU.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:GOMAXPROCS = "4"
$env:COMMITJL_CHECKPOINT_ROWS = "1"

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    $LogPath = Join-Path $env:COMMITJL_RESULTS_DIR "logs\$Name.log"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    Write-Host "[$(Get-Date -Format s)] $Stage / $Name"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($exitCode -ne 0) { throw "$Name failed (exit $exitCode); see $LogPath" }
}

function Select-ProofConfig {
    $RQ1 = Join-Path $RQ1Root "rq1_fidelity.parquet"
    if (-not (Test-Path -LiteralPath $RQ1)) { throw "Complete RQ1 results are required: $RQ1" }
    New-Item -ItemType Directory -Force -Path $ProofRoot | Out-Null
    $env:COMMITJL_RESULTS_DIR = $ProofRoot
    Invoke-PythonStep "select_proof_shapes" @(
        "-m", "experiments.select_proof_shapes", "--rq1", $RQ1,
        "--out", $SelectedConfig, "--max-shapes", "$MaxProofShapes"
    )
}

Push-Location $CodeRoot
try {
    switch ($Stage) {
        "AccuracyPilot" {
            New-Item -ItemType Directory -Force -Path $PilotRoot | Out-Null
            $env:COMMITJL_RESULTS_DIR = $PilotRoot
            Invoke-PythonStep "preflight" @("-m", "pytest", "tests", "--ignore=tests/test_zk.py", "-q")
            $env:COMMITJL_CHECKPOINT_ROWS = "8"
            Invoke-PythonStep "rq1_pilot" @("-m", "experiments.rq1_fidelity", "--config", "experiments/configs/accuracy_pilot.yaml")
            Invoke-PythonStep "accuracy_report" @("-m", "experiments.accuracy_report",
                "--results", (Join-Path $PilotRoot "rq1_fidelity.parquet"), "--out-dir", $PilotRoot)
            Invoke-PythonStep "pilot_table" @("-m", "experiments.make_tables", "--table", "fidelity-a")
            Invoke-PythonStep "pilot_figure" @("-m", "experiments.make_figures", "--fig", "fidelity-vs-m")
        }
        "PrecisionSweep" {
            New-Item -ItemType Directory -Force -Path $PrecisionRoot | Out-Null
            $env:COMMITJL_RESULTS_DIR = $PrecisionRoot
            $env:COMMITJL_CHECKPOINT_ROWS = "8"
            Invoke-PythonStep "precision" @("-m", "experiments.rq5_precision", "--config", "experiments/configs/paper_precision.yaml")
            Invoke-PythonStep "precision_table" @("-m", "experiments.make_tables", "--table", "precision-a")
            Invoke-PythonStep "precision_figure" @("-m", "experiments.make_figures", "--fig", "precision-sensitivity")
        }
        "FullNative" {
            & (Join-Path $PSScriptRoot "native.ps1") -CampaignName "native_serial" -GrindingWorkers $GrindingWorkers
            if (-not $?) { throw "Full native campaign failed" }
        }
        "RepresentativeProof" {
            Select-ProofConfig
            $env:COMMITJL_CHECKPOINT_ROWS = "8"
            Invoke-PythonStep "rq2_proof" @("-m", "experiments.rq2_proof_cost", "--config", $SelectedConfig)
        }
        "ProofAblation" {
            Select-ProofConfig
            New-Item -ItemType Directory -Force -Path $AblationRoot | Out-Null
            $env:COMMITJL_RESULTS_DIR = $AblationRoot
            $env:COMMITJL_CHECKPOINT_ROWS = "8"
            Invoke-PythonStep "ablation" @("-m", "experiments.ablation", "--config", $SelectedConfig)
        }
        "FinalReport" {
            New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
            $env:COMMITJL_RESULTS_DIR = $ReportRoot
            $inputs = @(
                @{ Root = $RQ1Root; Name = "rq1_fidelity.parquet" },
                @{ Root = $RQ3Root; Name = "rq3_overhead.parquet" },
                @{ Root = $RQ3Root; Name = "rq3_grinding.parquet" },
                @{ Root = $RQ4Root; Name = "rq4_certificates.parquet" },
                @{ Root = $RQ5Root; Name = "rq5_scaling.parquet" },
                @{ Root = $PrecisionRoot; Name = "rq5_precision.parquet" },
                @{ Root = $ProofRoot; Name = "rq2_proof_cost.parquet" },
                @{ Root = $AblationRoot; Name = "ablation.parquet" }
            )
            foreach ($input in $inputs) {
                $source = Join-Path $input.Root $input.Name
                if (-not (Test-Path -LiteralPath $source)) { throw "Missing paper result: $source" }
                Copy-Item -LiteralPath $source -Destination (Join-Path $ReportRoot $input.Name) -Force
            }
            Invoke-PythonStep "validate_inputs" @("-m", "experiments.validate_paper_results")
            Invoke-PythonStep "tables" @("-m", "experiments.make_tables", "--table", "all")
            Invoke-PythonStep "figures" @("-m", "experiments.make_figures", "--fig", "all")
            Invoke-PythonStep "headline" @("-m", "experiments.summarize")
        }
    }
    Write-Host "[$(Get-Date -Format s)] $Stage complete"
}
finally { Pop-Location }
