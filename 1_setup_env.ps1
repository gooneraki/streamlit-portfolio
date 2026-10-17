param(
    [string]$PythonPath = (Join-Path $env:USERPROFILE "scoop\apps\python314\current\python.exe")
)

$ErrorActionPreference = "Stop"

Write-Host "Checking Python and virtual environment..." -ForegroundColor Cyan

$venvPath = Join-Path $PSScriptRoot "venv"
$venvPythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$homePyPath = Join-Path $PSScriptRoot "Home.py"

if (-not (Test-Path -Path $PythonPath -PathType Leaf)) {
    throw "Scoop Python not found at '$PythonPath'. Install python314 or pass -PythonPath."
}

$pythonInfoCode = "import json, os, sys; print(json.dumps({'base': os.path.realpath(sys._base_executable), 'version': sys.version, 'is_venv': sys.prefix != sys.base_prefix}))"
$sourceInfoJson = & $PythonPath -c $pythonInfoCode
if ($LASTEXITCODE -ne 0) {
    throw "Failed to run the requested Python at '$PythonPath'."
}
$sourceInfo = $sourceInfoJson | ConvertFrom-Json
if ($sourceInfo.is_venv) {
    throw "PythonPath must point to a base Python installation, not another virtual environment."
}

if (Test-Path -Path $venvPath) {
    $recoveryMessage = "Close processes using venv, rename '$venvPath' as a backup, then rerun setup to create a fresh environment."
    if (-not (Test-Path -Path $venvPythonPath -PathType Leaf)) {
        throw "The existing virtual environment is incomplete. $recoveryMessage"
    }

    $venvInfoJson = & $venvPythonPath -c $pythonInfoCode
    if ($LASTEXITCODE -ne 0) {
        throw "The existing virtual environment cannot run. $recoveryMessage"
    }
    $venvInfo = $venvInfoJson | ConvertFrom-Json
    if (-not $venvInfo.is_venv -or $venvInfo.base -ine $sourceInfo.base -or $venvInfo.version -cne $sourceInfo.version) {
        throw "The existing virtual environment does not match '$PythonPath' (current base: $($venvInfo.base)). $recoveryMessage"
    }

    Write-Host "Reusing matching virtual environment: $venvPath" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment from: $PythonPath" -ForegroundColor Cyan
    & $PythonPath -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment at '$venvPath'."
    }
}

$activePythonVersion = & $venvPythonPath --version
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read the virtual environment's Python version."
}
$activePythonPath = & $venvPythonPath -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to run the virtual environment's Python."
}
$activePythonVersionNumber = ($activePythonVersion -replace '^Python\s+', '').Trim()
Write-Host "Active Python version: $activePythonVersion" -ForegroundColor Green
Write-Host "Active Python path: $activePythonPath" -ForegroundColor Green

# Step 3: Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip." -ForegroundColor Red
    exit 1
}

# Step 4: Install dependencies
Write-Host "Installing dependencies from custom_requirements.txt..." -ForegroundColor Cyan
& $venvPythonPath -m pip install -r (Join-Path $PSScriptRoot "custom_requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies." -ForegroundColor Red
    exit 1
}

# Step 5: Freeze dependencies
Write-Host "Freezing dependencies in requirements.txt..." -ForegroundColor Cyan
& $venvPythonPath -m pip freeze > (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to freeze dependencies." -ForegroundColor Red
    exit 1
}

# Step 5.1: Sync Home.py development version to the active venv version
Write-Host "Updating DEV_VERSION in Home.py..." -ForegroundColor Cyan
$homePyContent = Get-Content -Path $homePyPath -Raw
$versionPattern = '(?m)^(\s*DEV_VERSION\s*=\s*)["'']([^"'']+)["'']'
$versionMatch = [regex]::Match($homePyContent, $versionPattern)
if (-not $versionMatch.Success) {
    Write-Host "Failed to find DEV_VERSION in Home.py." -ForegroundColor Red
    exit 1
}
$updatedHomePyContent = [regex]::Replace(
    $homePyContent,
    $versionPattern,
    { param($match) "$($match.Groups[1].Value)`"$activePythonVersionNumber`"" },
    1
)
Set-Content -Path $homePyPath -Value $updatedHomePyContent -NoNewline
Write-Host "DEV_VERSION updated to $activePythonVersionNumber" -ForegroundColor Green

Write-Host "Setup complete." -ForegroundColor Green
