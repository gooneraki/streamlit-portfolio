# Script to update the virtual environment with the custom requirements

# Check if virtual environment exists
if (-Not (Test-Path -Path "./venv")) {
    Write-Host "Virtual environment not found. Please create it first." -ForegroundColor Red
    exit 1
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Reinstall custom requirements
Write-Host "Installing custom requirements..." -ForegroundColor Cyan
pip install -r custom_requirements.txt

# Freezing requirements
Write-Host "Freezing requirements..." -ForegroundColor Cyan
pip freeze > requirements.txt

Write-Host "Done." -ForegroundColor Green