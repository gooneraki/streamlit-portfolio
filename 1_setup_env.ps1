# Step 0-2: Resolve Python source, create venv if needed, activate, then display active version/path
Write-Host "Checking Python and virtual environment..." -ForegroundColor Cyan

$venvPath = Join-Path $PSScriptRoot "venv"
$venvPythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (Test-Path -Path $venvPythonPath) {
    Write-Host "Virtual environment already exists. Activating..." -ForegroundColor Yellow
} else {
    Write-Host "Virtual environment not found. Checking global Python..." -ForegroundColor Cyan
    $globalPythonPath = python -c "import sys; print(sys.executable)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "No virtual environment found and Python is not installed or not found in PATH." -ForegroundColor Red
        exit 1
    }

    Write-Host "Global Python path: $globalPythonPath" -ForegroundColor Green
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
if (-Not (Get-Command "deactivate" -ErrorAction SilentlyContinue)) {
    Write-Host "Failed to activate the virtual environment." -ForegroundColor Red
    exit 1
}

$activePythonVersion = python --version 2>&1
$activePythonPath = python -c "import sys; print(sys.executable)" 2>&1
Write-Host "Active Python version: $activePythonVersion" -ForegroundColor Green
Write-Host "Active Python path: $activePythonPath" -ForegroundColor Green

# Step 3: Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Step 4: Install dependencies
Write-Host "Installing dependencies from custom_requirements.txt..." -ForegroundColor Cyan
pip install -r custom_requirements.txt

# Step 5: Freeze dependencies
Write-Host "Freezing dependencies in requirements.txt..." -ForegroundColor Cyan
pip freeze > requirements.txt

# Step 6: Deactivate virtual environment
Write-Host "Deactivating virtual environment..." -ForegroundColor Cyan
deactivate

Write-Host "Setup complete." -ForegroundColor Green
