# Portfolio Historical Data Analysis

This repository contains a Streamlit-based application for analyzing historical financial data. The app fetches data from Yahoo Finance using the `yfinance` library and provides insights into individual assets and portfolio performance.

## Features

- **Single Asset Analysis**: Analyze the historical performance of a single asset, including trends and statistics.
- **Portfolio Analysis**: Evaluate the historical performance of an entire portfolio, with comprehensive stats and visualizations.
- **Currency and Cryptocurrency Information**: Fetch historical data for currency pairs and cryptocurrencies.

## Project Structure

```
Home.py
1_setup_env.ps1
2_run_streamlit_app.ps1
requirements.txt
pages/
    2_Symbol_View.py
    3_Currency_View.py
    4_Portfolio_View.py
utilities/
    constants.py
    go_charts.py
    utilities.py
```

### Key Files

- `Home.py`: The home page of the application.
- `pages/`: Contains individual pages for symbol, currency, and portfolio views.
- `utilities/`: Utility functions and constants used across the app.
- `requirements.txt`: Lists the Python dependencies for the project.
- `1_setup_env.ps1`: PowerShell script to set up the virtual environment and install dependencies.
- `2_run_streamlit_app.ps1`: PowerShell script to run the Streamlit app.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd streamlit-portfolio
   ```

2. Install Scoop's `python314`, then set up the virtual environment and install dependencies:
   ```powershell
   .\1_setup_env.ps1
   ```

   Setup uses `%USERPROFILE%\scoop\apps\python314\current\python.exe`, not
   `python` from Windows PATH. It creates `venv` only when missing and reuses an
   existing environment when its base interpreter and Python version match.
   It does not overwrite an existing environment's launcher. If the environment
   is broken or uses a different Python, setup stops with recovery instructions
   instead of modifying it. To use another installation:

   ```powershell
   .\1_setup_env.ps1 -PythonPath 'C:\path\to\python.exe'
   ```

   When setup reports a mismatch (including after a Python upgrade), close
   processes using `venv`, rename the folder as a backup, then rerun setup.
   Dependencies will be installed into a fresh environment; setup never deletes
   the old one for you.

   Setup also syncs `DEV_VERSION` in `Home.py` to the active Python version
   without adding trailing newlines.

3. Run the application:
   ```powershell
   .\2_run_streamlit_app.ps1
   ```

4. Open the app in your browser at `http://localhost:8501`.

### VS Code interpreter (this workspace only)

The workspace settings default to `venv\Scripts\python.exe` and enable terminal
environment activation. This environment is based on Scoop's `python314`; your
Windows default Python is unchanged.

If VS Code already remembers a different interpreter, run **Python: Select
Interpreter** from the Command Palette, choose **Enter interpreter path**, and
select `${workspaceFolder}\venv\Scripts\python.exe` (using the actual project
folder in the file picker). VS Code remembers that selection for this workspace;
`python.defaultInterpreterPath` only supplies the initial default.
Close existing terminals and open a new one after selecting the interpreter.

All three PowerShell scripts use the project's Python executable explicitly and
resolve project files relative to the script, so they do not rely on activation,
Windows PATH, or your terminal's working directory. Use `3_update_env.ps1` after
changing `custom_requirements.txt` to install dependencies and refresh
`requirements.txt`.

## Dependencies

The project uses the following Python libraries:

- `streamlit==1.45.0`
- `yfinance==0.2.57`
- `pandas==2.2.3`
- `numpy==2.2.5`
- `altair==5.5.0`
- `plotly==6.0.1`

## Disclaimer

This app is intended for informational purposes only. Historical data can help forecast future trends, but such predictions are inherently uncertain. The app is not responsible for any financial losses incurred based on its use.