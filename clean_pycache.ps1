# Script to delete all __pycache__ folders
Write-Host "Cleaning up __pycache__ folders..." -ForegroundColor Green

# Find all __pycache__ directories recursively
$pycacheFolders = Get-ChildItem -Path . -Name "__pycache__" -Recurse -Directory

if ($pycacheFolders.Count -eq 0) {
    Write-Host "No __pycache__ folders found." -ForegroundColor Yellow
} else {
    Write-Host "Found $($pycacheFolders.Count) __pycache__ folder(s):" -ForegroundColor Cyan
    
    $deletedCount = 0
    
    foreach ($folder in $pycacheFolders) {
        try {
            Remove-Item -Path $folder -Recurse -Force
            Write-Host "Deleted: $folder" -ForegroundColor Green
            $deletedCount++
        } catch {
            Write-Host "Failed to delete: $folder - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    Write-Host "`nCleanup complete! Deleted $deletedCount __pycache__ folder(s)." -ForegroundColor Green
} 