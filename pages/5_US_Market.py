
import streamlit as st
import yfinance as yf


@st.cache_data
def get_history(symbols: list[str], period: str):
    tickers = yf.Tickers(symbols)
    return tickers.history(period=period)


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
    "SPY"
]

# history = yf.download(symbols, period="10y")
history = get_history(symbols, "10y")
history_close = history.filter(like="Close")

st.dataframe(history_close.tail())
