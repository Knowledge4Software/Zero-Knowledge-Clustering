param(
    [string]$Plan = "C:\PyProjects\zero-knowledge\output\project_cleanup_20261107_plan.json"
)
$ErrorActionPreference = "Stop"
$taskPlanPath = [IO.Path]::GetFullPath($Plan)
$taskPlan = Get-Content -LiteralPath $taskPlanPath -Raw | ConvertFrom-Json
$taskWorkspace = [IO.Path]::GetFullPath($taskPlan.workspace).TrimEnd('\')
$taskAllowed = @($taskPlan.allowed_roots | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') })
$taskProtected = @($taskPlan.protected | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') })
$taskReceiptName = [IO.Path]::GetFileNameWithoutExtension($taskPlanPath).Replace("_plan", "") + "_deletions.json"
$taskReceipt = Join-Path (Join-Path $taskWorkspace "output") $taskReceiptName

function IsWithin([string]$Value, [string]$Parent) {
    return $Value.Equals($Parent, [StringComparison]::OrdinalIgnoreCase) -or
        $Value.StartsWith($Parent + '\', [StringComparison]::OrdinalIgnoreCase)
}

# Validate every absolute deletion target before deleting anything.
$taskTargets = @()
foreach ($taskEntry in $taskPlan.targets) {
    $taskTarget = [IO.Path]::GetFullPath($taskEntry.path).TrimEnd('\')
    if ($taskTarget.Equals($taskWorkspace, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing workspace-root deletion"
    }
    if (-not ($taskAllowed | Where-Object { IsWithin $taskTarget $_ })) {
        throw "Deletion target outside the workspace/user-named folders: $taskTarget"
    }
    foreach ($taskPin in $taskProtected) {
        if ((IsWithin $taskTarget $taskPin) -or (IsWithin $taskPin $taskTarget)) {
            throw "Deletion intersects protected content: $taskTarget"
        }
    }
    $taskParent = [IO.DirectoryInfo]([IO.Path]::GetDirectoryName($taskTarget))
    while ($null -ne $taskParent -and (IsWithin $taskParent.FullName $taskWorkspace)) {
        if ($taskParent.Exists -and ($taskParent.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Deletion parent is a link/junction: $($taskParent.FullName)"
        }
        $taskParent = $taskParent.Parent
    }
    $taskTargets += [pscustomobject]@{ path = $taskTarget; reason = $taskEntry.reason }
}

$taskDeleted = [Collections.Generic.List[object]]::new()
$taskFailures = [Collections.Generic.List[object]]::new()
foreach ($taskEntry in $taskTargets) {
    try {
        if (Test-Path -LiteralPath $taskEntry.path) {
            Remove-Item -LiteralPath $taskEntry.path -Recurse -Force -ErrorAction Stop
        }
        $taskDeleted.Add($taskEntry)
    } catch {
        $taskFailures.Add([pscustomobject]@{ path = $taskEntry.path; error = $_.Exception.Message })
    }
    if (($taskDeleted.Count + $taskFailures.Count) % 25 -eq 0) {
        [pscustomobject]@{ complete = $false; deleted = @($taskDeleted.ToArray()); failures = @($taskFailures.ToArray()) } |
            ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReceipt -Encoding UTF8
    }
}
[pscustomobject]@{ complete = ($taskFailures.Count -eq 0); deleted = @($taskDeleted.ToArray()); failures = @($taskFailures.ToArray()) } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskReceipt -Encoding UTF8
Write-Output ("Deleted targets: {0}; failed targets: {1}" -f $taskDeleted.Count, $taskFailures.Count)
if ($taskFailures.Count) {
    $taskFailures | Format-Table -AutoSize
    exit 1
}
