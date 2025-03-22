# Check if virtual environment exists
if (-Not (Test-Path -Path "./venv")) {
    Write-Host "Virtual environment not found. Please create it first." -ForegroundColor Red
    exit 1
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Run the Streamlit app
Write-Host "Running Streamlit app: 1_Home.py" -ForegroundColor Cyan
streamlit run 1_Home.py
