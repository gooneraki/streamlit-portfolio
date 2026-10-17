# Script to update the virtual environment with the custom requirements
$ErrorActionPreference = "Stop"
$venvPythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

# Check if virtual environment exists
if (-Not (Test-Path -Path $venvPythonPath -PathType Leaf)) {
    Write-Host "Virtual environment not found. Run 1_setup_env.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

# Reinstall custom requirements
Write-Host "Installing custom requirements..." -ForegroundColor Cyan
& $venvPythonPath -m pip install -r (Join-Path $PSScriptRoot "custom_requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install custom requirements."
}

# Freezing requirements
Write-Host "Freezing requirements..." -ForegroundColor Cyan
& $venvPythonPath -m pip freeze > (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to freeze requirements."
}

Write-Host "Done." -ForegroundColor Green