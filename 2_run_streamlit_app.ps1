$ErrorActionPreference = "Stop"
$venvPythonPath = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

if (-Not (Test-Path -Path $venvPythonPath -PathType Leaf)) {
    Write-Host "Virtual environment not found. Run 1_setup_env.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $venvPythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
& $venvPythonPath -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements."
}

# Run the Streamlit app
Write-Host "Running Streamlit app: Home.py" -ForegroundColor Cyan
Push-Location $PSScriptRoot
try {
    & $venvPythonPath -m streamlit run (Join-Path $PSScriptRoot "Home.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
