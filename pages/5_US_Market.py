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

# history = yf.download(symbols, period="10y")
symbol_data = fetch_symbol_data(symbols, "10y")
# history_close = history.filter(like="Close")

# st.dataframe(history_close.tail())

# for symbol in symbol_data["infos"]:
#     st.write(symbol["info"])


print(len(symbol_data["infos"]))
print(len(symbols))
# print(symbol_data["infos"][-2])

# print(symbol_data["history"])


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
