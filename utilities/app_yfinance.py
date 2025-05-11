""" Yfinance utilities for Streamlit app """
import yfinance as yf
import streamlit as st
import pandas as pd

valid_periods = ['1d', '5d', '1mo', '3mo',
                 '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

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


@st.cache_data
def retrieve_sector_industry_keys():
    """ Get the sector and industry keys """
    return list(yf.const.SECTOR_INDUSTY_MAPPING.keys())


@st.cache_data
def sector_yf(sector_key: str):
    """ Get the sector info """
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


@st.cache_data
def search_yf(search_input: str):
    """ Search for a symbol """
    try:
        search_results = yf.Search(search_input)
        return search_results.all
    except Exception as err:
        error_message = f"Error retrieving search results for '{search_input}': {err}"
        return error_message


@st.cache_data
def ticker_yf(symbol: str, info: bool, history: bool):
    """ Fetch the asset info for a given symbol """
    if info is False and history is False:
        error_message = "Both info and history cannot be False."
        return error_message
    try:
        result = {}

        ticker = yf.Ticker(symbol)

        if info:
            result['info'] = ticker.info
        if history:
            result['history'] = ticker.history(
                period='max', auto_adjust=True)['Close']

        return result
    except Exception as err:
        error_message = f"Error retrieving ticker details for '{symbol}': {err}"
        return error_message


def ticker_yf_info(symbol: str):
    """ Fetch the asset info for a given symbol """
    ticker = ticker_yf(symbol, info=True, history=False)

    return ticker['info'] if isinstance(ticker, dict) else ticker


def ticker_yf_history(symbol: str):
    """ Fetch the asset info for a given symbol """
    ticker_history = ticker_yf(symbol, info=False, history=True)['history']

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = ticker_history.index.tz_localize(None)

    ticker_history = ticker_history.resample('D').ffill()

    ticker_history = pd.concat([ticker_history, ticker_history.pct_change(365)], axis=1, keys=[
                               'value', 'annual_value_return'])

    return ticker_history


def fetch_fx_rate_history(asset_currency: str, base_currency: str):
    """ Fetch the fx rate for a given currency pair """

    if asset_currency == base_currency:
        return pd.Series(1, index=pd.date_range(start='1950-01-01', end=pd.Timestamp.today(), freq='D'))

    currency_ticker_name = asset_currency + base_currency + "=X"
    fx_ticker = yf.Ticker(currency_ticker_name)
    fx_history = fx_ticker.history(period='max')['Close']
    fx_history = fx_history.resample('D').ffill()
    fx_history.index = fx_history.index.tz_localize(None)

    return fx_history


@st.cache_data
def get_fx_history_2(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""
    currency_symbol = target_currency + base_currency + "=X"
    crypto_symbol = target_currency + "-" + base_currency

    currency_rate_history = yf.Ticker(
        currency_symbol).history(period='max')['Close']
    crypto_rate_history = yf.Ticker(
        crypto_symbol).history(period='max')['Close']

    return currency_rate_history if not currency_rate_history.empty else crypto_rate_history, currency_symbol if not currency_rate_history.empty else crypto_symbol


@st.cache_data
def market_yf(market: str):
    """ Fetch the market info for a given market """
    market_data = yf.Market(market)

    return {
        'summary': market_data.summary,
        'status': market_data.status
    }
