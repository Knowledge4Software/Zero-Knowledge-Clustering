param([Parameter(Mandatory=$true)][string]$PlanPath)
$ErrorActionPreference = 'Stop'
$Workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$WorkspacePrefix = $Workspace + '\'
$PlanFile = (Resolve-Path -LiteralPath $PlanPath).Path
if (-not $PlanFile.StartsWith($WorkspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Plan is outside workspace' }
$Plan = Get-Content -LiteralPath $PlanFile -Raw | ConvertFrom-Json
if ($Plan.workspace.TrimEnd('\') -ne $Workspace) { throw 'Plan workspace differs' }
$Protected = @($Plan.protected_prefixes | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') })
foreach ($Property in $Plan.protected_sha256.PSObject.Properties) {
    $Target = Join-Path $Workspace $Property.Name
    if ((Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Property.Value) { throw "Protected source changed: $Target" }
}
# Validate every resolved target before the first recursive removal.
foreach ($Row in $Plan.records) {
    $Target = [IO.Path]::GetFullPath($Row.absolute_path).TrimEnd('\')
    if ($Target -eq $Workspace -or -not $Target.StartsWith($WorkspacePrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe target: $Target" }
    foreach ($Keep in $Protected) {
        if ($Target -eq $Keep -or $Target.StartsWith($Keep+'\', [StringComparison]::OrdinalIgnoreCase) -or $Keep.StartsWith($Target+'\', [StringComparison]::OrdinalIgnoreCase)) { throw "Protected overlap: $Target" }
    }
    foreach ($Property in $Plan.protected_sha256.PSObject.Properties) {
        $Keep = [IO.Path]::GetFullPath((Join-Path $Workspace $Property.Name))
        if ($Target -eq $Keep -or $Keep.StartsWith($Target+'\', [StringComparison]::OrdinalIgnoreCase)) { throw "Pinned evidence overlap: $Target" }
    }
    if (Test-Path -LiteralPath $Target) {
        $Resolved = (Resolve-Path -LiteralPath $Target).Path
        if ($Resolved -ne $Target) { throw "Resolved path differs: $Target" }
        $Ancestor = Get-Item -LiteralPath $Target -Force
        while ($Ancestor.FullName -ne $Workspace) {
            if ($Ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Linked target: $Target" }
            $Ancestor = if ($Ancestor -is [IO.DirectoryInfo]) { $Ancestor.Parent } else { $Ancestor.Directory }
            if ($null -eq $Ancestor) { throw "No workspace ancestor: $Target" }
        }
    }
}
$ReceiptPath = Join-Path $Workspace 'output/project_cleanup_20261027.json'
$Results = [Collections.Generic.List[object]]::new()
$Counter = 0
$Started = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
function Save-Receipt([string]$State) {
    $Receipt = [ordered]@{ status=$State; started_unix=$Started; updated_unix=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();
        plan_sha256=(Get-FileHash -LiteralPath $PlanFile -Algorithm SHA256).Hash.ToLowerInvariant();
        deletion_not_archival=$true; completed_targets=$Results.Count; planned_targets=$Plan.records.Count; records=$Results.ToArray() }
    $Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
}
Save-Receipt 'deleting'
foreach ($Row in $Plan.records) {
    $Target = $Row.absolute_path
    if (Test-Path -LiteralPath $Target) {
        try {
            if ($Row.is_directory) { Remove-Item -LiteralPath $Target -Recurse -Force }
            else { Remove-Item -LiteralPath $Target -Force }
            if (Test-Path -LiteralPath $Target) { throw 'Target remains after removal' }
            $Results.Add([ordered]@{path=$Row.path; status='deleted'; files=$Row.files; bytes=$Row.bytes; reason=$Row.reason})
        } catch {
            $Results.Add([ordered]@{path=$Row.path; status='failed'; error=$_.Exception.Message})
            Save-Receipt 'failed'; throw
        }
    } else { $Results.Add([ordered]@{path=$Row.path; status='already_absent'; files=0; bytes=0}) }
    $Counter++
    if ($Counter % 25 -eq 0) { Save-Receipt 'deleting'; Write-Output "Deleted targets: $Counter / $($Plan.records.Count)" }
}
Save-Receipt 'deleted_awaiting_validation'
Write-Output "Deletion complete: $Counter targets. Receipt: $ReceiptPath"
