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

2. Set up the virtual environment and install dependencies:
   ```powershell
   .\1_setup_env.ps1
   ```

3. Run the application:
   ```powershell
   .\2_run_streamlit_app.ps1
   ```

4. Open the app in your browser at `http://localhost:8501`.

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