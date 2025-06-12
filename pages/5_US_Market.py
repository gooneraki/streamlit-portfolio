# pylint: disable=invalid-name
"""
This page is used to fetch the data for the US market
"""
import time
from typing import Any, Literal
import numpy as np
from utilities.utilities import get_exp_fitted_data
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf


print(f"\n--- US Market Page loaded at {time.strftime('%H:%M:%S')} ---\n")

st.set_page_config(
    page_title="US Market",
    page_icon=":bar_chart:",
    layout="centered")


@st.cache_data
def fetch_symbol_data(p_symbols: list[str], period: str):
    """
    Fetch symbol data from yfinance
    """
    tickers = yf.Tickers(p_symbols)
    ticker_list: list[dict[Literal['symbol', 'info'], Any]] = [{
        "symbol": symbol,
        'info': yf.Ticker(symbol).info} for symbol in p_symbols]

    return {
        "history": tickers.history(period=period),
        "infos": ticker_list
    }


STOCK_SYMBOLS = [
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLK",
    "XLB",
    "XLRE",
    "XLU",
    "SPY",
    "KOKO"
]

CURRENCY_SYMBOLS = [
    "EURUSD=X",  # Canadian Dollar to US Dollar
]

symbol_data = fetch_symbol_data(STOCK_SYMBOLS, "10y")


def extract_invalid_symbols(p_symbol_data: dict):
    """
    Extract invalid symbols from the symbol data
    """
    return [data["symbol"] for data in p_symbol_data["infos"] if data["info"] is None or "symbol" not in data["info"]]


invalid_symbols = extract_invalid_symbols(symbol_data)


def remove_invalid_symbols(p_symbol_data: dict, p_invalid_symbols: list[str]):
    """
    Remove invalid symbols from the symbol data
    """
    return {
        "history": p_symbol_data["history"].loc[
            :,
            ~p_symbol_data["history"].columns.get_level_values(1).isin(p_invalid_symbols)],
        "infos": [data for data in p_symbol_data["infos"] if data["symbol"] not in p_invalid_symbols]
    }


valid_symbol_data = remove_invalid_symbols(symbol_data, invalid_symbols)


history_close: pd.DataFrame = valid_symbol_data["history"]["Close"].copy()
tickers_info = valid_symbol_data["infos"]


infos_df = pd.DataFrame(
    [(ticker_info['symbol'], ticker_info['info'].get('longName', 'No short name available'))
        for ticker_info in tickers_info],
    columns=['Symbol', 'Short Name'])
st.dataframe(infos_df, use_container_width=True, hide_index=True)


history_close.dropna(axis=0, how='any', inplace=True)
# print('History Close Data:')
# print(history_close)

first_date = history_close.index[0]
last_date = history_close.index[-1]
days_duration = (last_date - first_date).days + 1
years_duration = days_duration / 365.25
total_points = history_close.shape[0]
points_per_year = total_points / years_duration
print(f"First date: {first_date}")
print(f"Last date: {last_date}")
print(f"Days duration: {days_duration}")
print(f"Years duration: {years_duration}")
print(f"Total points: {total_points}")
print(f"Points per year: {points_per_year}")


fig, ax = plt.subplots()
history_close.plot(ax=ax, title="US Market Historical Close Prices")
ax.set_ylabel("Price (USD)")
ax.set_xlabel("Date")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
st.pyplot(fig)

history_close_normalized = history_close.copy()
history_close_normalized = history_close_normalized.div(
    history_close_normalized.iloc[0, :], axis=1).mul(100)

fig, ax = plt.subplots()
history_close_normalized.plot(ax=ax, title="US Market Normalized Close Prices")
ax.set_ylabel("Normalized Price")
ax.set_xlabel("Date")
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
st.pyplot(fig)


history_close_pct = history_close.pct_change().dropna()


mean_returns = history_close_pct.describe().T


mean_returns['annualized_mean'] = mean_returns['mean'] * points_per_year
mean_returns['annualized_std'] = mean_returns['std'] * (
    points_per_year ** 0.5)
mean_returns['annualized_sharpe'] = mean_returns['annualized_mean'] / \
    mean_returns['annualized_std']


# scatter plot of annualized mean vs annualized std
fig, ax = plt.subplots()
mean_returns.plot.scatter(
    x='annualized_std', y='annualized_mean', ax=ax, title="Annualized Mean vs Annualized Std")
ax.set_xlabel("Annualized Std")
ax.set_ylabel("Annualized Mean")

for i, row in mean_returns.iterrows():
    ax.annotate(row.name,
                (row['annualized_std'] + 0.002, row['annualized_mean']+0.002))

st.pyplot(fig)


# Covariance and correlation matrices
# cov_matrix = history_close_pct.cov()

# corr_matrix = history_close_pct.corr()

# # Heatmap of the correlation matrix
# fig, ax = plt.subplots()
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# ax.set_title("Correlation Matrix of US Market Stocks")
# st.pyplot(fig)


# Logarithmic returns
log_returns = history_close.apply(
    lambda x: np.log(x / x.shift(1))).dropna()

log_returns_description = log_returns.describe().T
# log_returns_description['first'] = history_close.iloc[0, :]
# log_returns_description['last'] = history_close.iloc[-1, :]

# log_returns_description['validation'] = log_returns_description['first'] * \
#     np.exp(log_returns_description['mean'] *
#            history_close_log_returns.index.size)

log_returns_description['annualized_mean'] = log_returns_description['mean'] * points_per_year
log_returns_description['annualized_std'] = log_returns_description['std'] * \
    (points_per_year ** 0.5)
log_returns_description['annualized_sharpe'] = log_returns_description['annualized_mean'] / \
    log_returns_description['annualized_std']


print('history_close_description')
# history_close_description['first'] = history_close.iloc[0, :]
# history_close_description['last'] = history_close.iloc[-1, :]
# history_close_description['validation'] = history_close_description['first'] * ((1 +
#                                                                                  history_close_description['annualized_mean']) ** years_duration)
print(mean_returns)


print('Log Returns Description:')
print(log_returns_description)

# cagr_returns = log_returns_description.copy()
# cagr_returns['cagr'] = (history_close.iloc[-1, :] /
#                         history_close.iloc[0, :]) ** (1 / years_duration) - 1
# cagr_returns = cagr_returns[['cagr']].copy()
# print('CAGR Returns:')
# print(cagr_returns)


history_close_fitted = history_close.apply(get_exp_fitted_data)
# print('Fitted Data:')
# print(history_close_fitted)

squared_errors = ((history_close / history_close_fitted)-1) ** 2
# print('Squared Errors:')
# print(squared_errors)

cagr_metrics = np.sqrt(squared_errors.mean())
cagr_metrics = pd.DataFrame(cagr_metrics, columns=['RMSE'])
cagr_metrics['CAGR'] = (history_close.iloc[-1, :] /
                        history_close.iloc[0, :]) ** (1 / years_duration) - 1
cagr_metrics['CAGR_Fitted'] = (history_close_fitted.iloc[-1, :] /
                               history_close_fitted.iloc[0, :]) ** (1 / years_duration) - 1

cagr_metrics['CAGR-to-RMSE Ratio'] = cagr_metrics['CAGR'] / \
    cagr_metrics['RMSE']
cagr_metrics['Over/Under Today'] = history_close_fitted.iloc[-1, :] / \
    history_close.iloc[-1, :] - 1

cagr_metrics['Log Returns (annualised)'] = log_returns_description['annualized_mean']
cagr_metrics['Log Returns Std (annualised)'] = log_returns_description['annualized_std']
cagr_metrics['Log Returns Sharpe (annualised)'] = log_returns_description['annualized_sharpe']

cagr_metrics['First Date'] = history_close.index[0]
cagr_metrics['Last Date'] = history_close.index[-1]
cagr_metrics['Years Duration'] = years_duration

print('CAGR Metrics:')
print(cagr_metrics)
st.dataframe(cagr_metrics.style.format({
    "CAGR": "{:.1%}",
    "CAGR_Fitted": "{:.1%}",
    "RMSE": "{:.1%}",
    "Over/Under Today": "{:.1%}",
    "Log Returns (annualised)": "{:.1%}",
    "Log Returns Std (annualised)": "{:.1%}",
    "Log Returns Sharpe (annualised)": "{:.2f}",
    "Years Duration": "{:.2f}",
    "First Date": lambda x: x.strftime('%Y-%m-%d'),
    "Last Date": lambda x: x.strftime('%Y-%m-%d')
}),
    use_container_width=True,

)


monthly_close = history_close.resample('ME').last()
