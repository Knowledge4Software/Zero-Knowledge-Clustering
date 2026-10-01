param(
    [ValidateSet("All", "RQ3", "RQ4", "RQ5Gamma", "RQ5Precision", "Analysis")]
    [string]$Stage = "All",
    [ValidateRange(1, 8)]
    [int]$Workers = 6
)

$ErrorActionPreference = "Stop"
$CodeRoot = Split-Path -Parent $PSScriptRoot
$ResultRoot = Join-Path $CodeRoot "results"
$LogRoot = Join-Path $ResultRoot "overview\runs\logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:COMMITJL_CHECKPOINT_ROWS = "1"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:GOMAXPROCS = "4"

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    $log = Join-Path $LogRoot "$Name.log"
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
    if ($Stage -ne "Analysis") {
        $stages = if ($Stage -eq "All") { @("RQ3", "RQ4", "RQ5Gamma", "RQ5Precision") } else { @($Stage) }
        foreach ($item in $stages) {
            $relative = switch ($item) {
                "RQ3" { "rq3\runs\paired" }
                "RQ4" { "rq4\runs\real" }
                "RQ5Gamma" { "rq5\runs\gamma" }
                "RQ5Precision" { "rq5\runs\precision_24ref" }
            }
            $output = Join-Path $ResultRoot $relative
            $LogRoot = Join-Path $output "logs"
            Invoke-Step "supplement_$item" @("-m", "campaigns.supplements", "--stage", $item,
                                               "--workers", "$Workers", "--out-dir", $output)
        }
    }
    $LogRoot = Join-Path $ResultRoot "overview\runs\logs"
    Invoke-Step "analysis" @("-m", "analysis.paper_results")
    Invoke-Step "author_overview" @("-m", "analysis.author_overview")
    Write-Host "[$(Get-Date -Format s)] $Stage complete. Per-RQ results: $ResultRoot; overview: $(Join-Path $ResultRoot 'overview\analysis')"
}
finally { Pop-Location }
