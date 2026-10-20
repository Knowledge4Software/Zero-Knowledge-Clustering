$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$archiveRoot = Join-Path $projectRoot 'paper_history/before_three_versions_20261020'
if (Test-Path -LiteralPath $archiveRoot) { throw "Archive already exists: $archiveRoot" }

function Assert-WorkspacePath([string] $path) {
    $resolved = [System.IO.Path]::GetFullPath($path)
    if (-not $resolved.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside workspace: $resolved"
    }
    return $resolved
}

New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$entries = [System.Collections.Generic.List[object]]::new()
foreach ($relative in @('manuscript.tex', 'references.bib', 'README.md', 'paper_history/README.md', 'paper_results/README.md', 'paper_results/reuse_revision')) {
    $source = Assert-WorkspacePath (Join-Path $projectRoot $relative)
    $destination = Assert-WorkspacePath (Join-Path $archiveRoot ('working_source/' + $relative))
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse
}

$templateRoot = Assert-WorkspacePath (Join-Path $projectRoot 'paper_assets')
New-Item -ItemType Directory -Force -Path $templateRoot | Out-Null
foreach ($name in @('sn-jnl.cls', 'sn-mathphys-num.bst')) {
    $destination = Join-Path $templateRoot $name
    if (Test-Path -LiteralPath $destination) { throw "Template already exists: $destination" }
    Copy-Item -LiteralPath (Join-Path $projectRoot ('Sparse_JL_for_ZK_Clustering/' + $name)) -Destination $destination
}

$moves = @(
    'Sparse_JL_for_ZK_Clustering',
    'output/commitjl_manuscript_revised_sources.zip',
    'output/manuscript_revision_validation.json',
    'output/reusable_statistics_20261019_sources.zip',
    'output/reusable_statistics_20261019_validation.json',
    'output/pdf/commitjl_manuscript_revised.pdf',
    'output/pdf/reusable_statistics_20261019.pdf'
)
foreach ($relative in $moves) {
    $source = Assert-WorkspacePath (Join-Path $projectRoot $relative)
    if (-not (Test-Path -LiteralPath $source)) { continue }
    $destination = Assert-WorkspacePath (Join-Path $archiveRoot ('superseded/' + $relative))
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    $files = if ((Get-Item -LiteralPath $source).PSIsContainer) { @(Get-ChildItem -LiteralPath $source -File -Recurse) } else { @(Get-Item -LiteralPath $source) }
    foreach ($file in $files) {
        $entries.Add([pscustomobject]@{ original = $file.FullName.Substring($projectRoot.Length + 1); sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower(); bytes = $file.Length })
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
}

$record = [pscustomobject]@{ date = '2026-10-20'; archive = $archiveRoot; deleted_files = 0; experiment_results_moved = 0; entries = $entries }
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $archiveRoot 'organization_manifest.json') -Encoding UTF8
Write-Output "Archived $($entries.Count) superseded paper files. No files deleted. Experiment evidence retained in place."
