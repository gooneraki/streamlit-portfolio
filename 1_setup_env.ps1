# Step 1: Check if 'venv' folder exists
if (-Not (Test-Path -Path "./venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists. Skipping creation." -ForegroundColor Yellow
}

# Step 2: Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
if (-Not (Get-Command "deactivate" -ErrorAction SilentlyContinue)) {
    Write-Host "Failed to activate the virtual environment." -ForegroundColor Red
    exit 1
}

# Step 3: Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Step 4: Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

# Step 5: Deactivate virtual environment
Write-Host "Deactivating virtual environment..." -ForegroundColor Cyan
deactivate

Write-Host "Setup complete." -ForegroundColor Green
