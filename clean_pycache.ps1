# Write a powershell script to clean all __pycache__ folders

Write-Host "Cleaning __pycache__ folders and .pyc files from specific directories..." -ForegroundColor Green

# Get the current directory
$currentDir = Get-Location

# Define the directories to search in
$searchDirs = @("classes", "pages", "utilities")

# Find and remove all __pycache__ directories
$pycacheDirs = @()
foreach ($dir in $searchDirs) {
    Write-Host "Searching in directory: $dir" -ForegroundColor Cyan
    $dirPath = Join-Path $currentDir $dir
    if (Test-Path $dirPath) {
        $found = Get-ChildItem -Path $dirPath -Recurse -Directory -Name "__pycache__"
        foreach ($foundDir in $found) {
            $pycacheDirs += Join-Path $dir $foundDir
        }
    } else {
        Write-Host "  Directory '$dir' not found" -ForegroundColor Yellow
    }
}

if ($pycacheDirs) {
    Write-Host "Found $($pycacheDirs.Count) __pycache__ directories:" -ForegroundColor Yellow
    foreach ($dir in $pycacheDirs) {
        $fullPath = Join-Path $currentDir $dir
        Write-Host "  Removing: $fullPath" -ForegroundColor Red
        Remove-Item -Path $fullPath -Recurse -Force
    }
} else {
    Write-Host "No __pycache__ directories found in specified directories." -ForegroundColor Yellow
}

# Find and remove all .pyc files
$pycFiles = @()
foreach ($dir in $searchDirs) {
    Write-Host "Searching for .pyc files in directory: $dir" -ForegroundColor Cyan
    $dirPath = Join-Path $currentDir $dir
    if (Test-Path $dirPath) {
        $found = Get-ChildItem -Path $dirPath -Recurse -File -Filter "*.pyc"
        $pycFiles += $found
    } else {
        Write-Host "  Directory '$dir' not found" -ForegroundColor Yellow
    }
}

if ($pycFiles) {
    Write-Host "Found $($pycFiles.Count) .pyc files:" -ForegroundColor Yellow
    foreach ($file in $pycFiles) {
        Write-Host "  Removing: $($file.FullName)" -ForegroundColor Red
        Remove-Item -Path $file.FullName -Force
    }
} else {
    Write-Host "No .pyc files found in specified directories." -ForegroundColor Yellow
}

Write-Host "Cleanup completed!" -ForegroundColor Green
