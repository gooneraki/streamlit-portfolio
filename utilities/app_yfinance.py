""" Yfinance utilities for Streamlit app """
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Union, TypedDict


YF_SECTOR_KEYS = list(yf.const.SECTOR_INDUSTY_MAPPING.keys())

# ######### #
# yf.Sector #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


@st.cache_data
def sector_yf(sector_key: str):
    """ Fetch the sector data """
    try:
        sector = yf.Sector(sector_key)

        return {
            'name': sector.name,
            'overview': sector.overview,
            'top_companies': sector.top_companies,
            'top_etfs': sector.top_etfs
        }

    except Exception as err:
        error_message = f"Error retrieving sector details for '{sector_key}': {err}"
        return error_message

# ^^^^^^^^^ #
# yf.Sector #
# ######### #

# ######### #
# yf.Search #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


@st.cache_data
def search_yf(search_input: str):
    """ Search for a symbol """
    try:
        search_results = yf.Search(search_input)
        return search_results.all
    except Exception as err:
        error_message = f"Error retrieving search results for '{search_input}': {err}"
        return error_message

# ^^^^^^^^^ #
# yf.Search #
# ######### #

# ######### #
# yf.Ticker #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


@st.cache_data
def yf_ticket_info(symbol: str):
    """ Fetch the asset info for a given symbol """
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info
    except Exception as err:
        return f"Error retrieving ticker details for '{symbol}': {err}"


@st.cache_data
def yf_ticket_history(symbol: str, period='max'):
    """ Fetch the asset history for a given symbol """
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, auto_adjust=True)['Close']
    except Exception as err:
        return f"Error retrieving ticker details for '{symbol}': {err}"


@st.cache_data
def get_fx_history(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""

    if base_currency == target_currency:
        return pd.Series(1, index=pd.date_range(start='1980-01-01', end=pd.Timestamp.today(), freq='D'))

    fx_symbol = target_currency + base_currency + "=X"
    fx_history = yf_ticket_history(fx_symbol)

    if not isinstance(fx_history, str):
        return fx_history

    crypto_symbol = target_currency + "-" + base_currency

    crypto_history = yf_ticket_history(crypto_symbol)

    if not isinstance(crypto_history, str):
        return crypto_history

    return f"Error retrieving historical data for either '{fx_symbol}' or '{crypto_symbol}'"

# ^^^^^^^^^ #
# yf.Ticker #
# ######### #


# ######### #
# yf.Market #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


@st.cache_data
def market_yf(market: str):
    """ Fetch the market info for a given market """
    try:
        market_data = yf.Market(market)

        return {
            'summary': market_data.summary,
            'status': market_data.status
        }
    except Exception as err:
        error_message = f"Error retrieving market details for '{market}': {err}"
        return error_message
# ^^^^^^^^^ #
# yf.Market #
# ######### #

# ######### #
# yf.Tickers #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


class TickersData(TypedDict):
    """Type definition for successful tickers data response"""
    history: pd.DataFrame
    news: dict[str, list[dict]]


@st.cache_data
def tickers_yf(symbols: list[str]) -> Union[str, TickersData]:
    """ 
    Fetch the tickers info for a given symbols

    Returns:
        Union[str, TickersData]: Either an error string or a dictionary with:
            - history: pd.DataFrame - Historical close prices
            - news: dict[str, list[dict]] - News articles
    """
    try:
        tickers = yf.Tickers(symbols)
        return {
            "history": tickers.history(period='max', auto_adjust=True)["Close"],
            "news": tickers.news()
        }
    except Exception as err:
        error_message = f"Error retrieving tickers details for '{symbols}': {err}"
        return error_message
# ^^^^^^^^^ #
# yf.Tickers #
# ######### #


# valid_periods = ['1d', '5d', '1mo', '3mo',
#                  '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


# def is_yf_type(value):
#     """ Check if the value is a yfinance type """
#     return 'yfinance' in str(type(value))


# def extract_private_attributes(instance: object):
#     """ Extract private attributes from a given instance """

#     result = {}

#     for key in [el for el in dir(instance) if not el.startswith('_')]:
#         try:

#             value = getattr(instance, key)

#             if not is_yf_type(value):
#                 result[key] = value
#             else:
#                 print(f"Skipping yfinance type: {key} - {value}")
#         except Exception:
#             continue

#     return result
