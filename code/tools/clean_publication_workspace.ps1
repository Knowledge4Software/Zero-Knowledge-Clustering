param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$archiveRoot = [System.IO.Path]::GetFullPath('C:/PyProjects/zero-knowledge-archive/cleanup-publication-20261020')
$retire = @(
    'paper_history/before_reuse_revision_20261019',
    'paper_history/before_review_controls_20261020',
    'paper_history/before_review_results_followup_20261020',
    'paper_history/before_three_versions_20261020',
    'paper_results/review_controls/generated_followup_full_20261020',
    'paper_results/review_controls/generated_followup_matched_20261020',
    'code/validation/outputs/review_certified_capture_followup_20261020',
    'code/validation/outputs/review_certified_capture_n1024_20261020',
    'code/validation/outputs/review_full_capture_20261020_0330',
    'code/validation/outputs/review_final_checks_20261020',
    'code/validation/outputs/review_followup_results_checks_20261020.json',
    'code/validation/outputs/review_followup_final_checks_20261020.json',
    'code/validation/outputs/review_revision_checks_20261020.json',
    'code/validation/outputs/review_live_status_20261020T052855Z.txt',
    'code/validation/outputs/review_results_status_20261020T055147Z.txt',
    'code/validation/outputs/audit_results.json',
    'output/Neurocomputing_readiness_20261020.md',
    'output/review_controls_status_20261020.md',
    'output/mobin_completed_results_20261020.md',
    'output/reusable_statistics_review_controls_20261020_draft_sources.zip',
    'output/reusable_statistics_review_controls_20261020_draft_validation.json',
    'output/reusable_statistics_review_results_20261020_draft_sources.zip',
    'output/reusable_statistics_review_results_20261020_draft_validation.json',
    'output/pdf/reusable_statistics_review_controls_20261020_draft.pdf',
    'output/pdf/reusable_statistics_review_results_20261020_draft.pdf',
    'code/results/reports'
)
$entries = @()
foreach ($relative in $retire) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $workspace $relative))
    $destination = [System.IO.Path]::GetFullPath((Join-Path $archiveRoot $relative))
    if (!$source.StartsWith($workspace + [System.IO.Path]::DirectorySeparatorChar)) { throw "Outside workspace: $source" }
    if (!$destination.StartsWith($archiveRoot + [System.IO.Path]::DirectorySeparatorChar)) { throw "Outside archive: $destination" }
    if (!(Test-Path -LiteralPath $source)) { continue }
    if (Test-Path -LiteralPath $destination) { throw "Archive target exists: $destination" }
    $item = Get-Item -LiteralPath $source
    $files = if ($item.PSIsContainer) { @(Get-ChildItem -LiteralPath $source -File -Recurse) } else { @($item) }
    foreach ($file in $files) {
        $fileRelative = $file.FullName.Substring($workspace.Length + 1).Replace('\','/')
        $entries += [ordered]@{path=$fileRelative; bytes=$file.Length; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower()}
    }
    if ($Apply) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Move-Item -LiteralPath $source -Destination $destination
    }
}
$receipt = [ordered]@{
    applied=[bool]$Apply; archive=$archiveRoot; files=$entries;
    note='Superseded reports, intermediate paper exports and duplicate evidence views archived outside the workspace. No raw measurement campaign or preserved supervisor source was deleted.'
}
if ($Apply) {
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $workspace 'code/validation/outputs/cleanup_publication_20261020.json') -Encoding UTF8
}
$totalBytes = 0
foreach ($entry in $entries) { $totalBytes += $entry['bytes'] }
Write-Output ("Archived={0}; files={1}; bytes={2}" -f [bool]$Apply, $entries.Count, $totalBytes)
