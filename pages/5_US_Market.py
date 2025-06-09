# pylint: disable=invalid-name
"""
This page is used to fetch the data for the US market
"""
import time
import streamlit as st
import yfinance as yf

print(f"\n US Market Page loaded at {time.strftime('%H:%M:%S')}")


@st.cache_data
def fetch_symbol_data(p_symbols: list[str], period: str):
    """
    Fetch symbol data from yfinance
    """
    tickers = yf.Tickers(p_symbols)
    ticker_list = [{
        "symbol": symbol,
        'info': yf.Ticker(symbol).info} for symbol in p_symbols]

    # print(ticker_list)

    return {
        "history": tickers.history(period=period),
        "infos": ticker_list
    }


symbols = [
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
    "EURUSD=X",
    "KOKO"
]

symbol_data = fetch_symbol_data(symbols, "10y")


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


history_close = valid_symbol_data["history"]["Close"].copy()

# remove nan
history_close.dropna(axis=0, how='any', inplace=True)
print(history_close)

# history_close.columns = history_close.columns.get_level_values(1)
st.dataframe(history_close)
