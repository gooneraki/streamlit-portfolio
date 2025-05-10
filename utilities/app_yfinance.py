""" Yfinance utilities for Streamlit app """
import yfinance as yf
import streamlit as st
import pandas as pd


@st.cache_data
def get_sector_details(sector_key: str):
    """ Get the sector info """
    try:
        sector = yf.Sector(sector_key)

        keys = [key[1:]
                for key in sector.__dict__ if key.startswith('_')]

        result = {}
        for key in keys:
            try:
                value = getattr(sector, key)
                result[key] = value
            except Exception:
                continue

        return result

    except Exception as err:
        error_message = f"Error retrieving sector details for '{sector_key}': {err}"
        return error_message


@st.cache_data
def retrieve_sector_industry_keys():
    """ Get the sector and industry keys """
    return list(yf.const.SECTOR_INDUSTY_MAPPING.keys())


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
def fetch_asset_info_2(symbol: str):
    """ Fetch the asset info for a given symbol """
    y_finance_ticker = yf.Ticker(symbol)
    return y_finance_ticker.info


@st.cache_data
def fetch_asset_history_2(symbol: str):
    """ Fetch the asset info for a given symbol """

    valid_periods = ['1d', '5d', '1mo', '3mo',
                     '6mo', '1y', '2y', '5y', '10y',  'max']
    y_finance_ticker = yf.Ticker(symbol)

    # Try all valid_periods (in reverse) and break if data is found
    for period in valid_periods[::-1]:
        ticker_history = y_finance_ticker.history(
            period=period, auto_adjust=True)['Close']
        if ticker_history.shape[0] > 0:
            break

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = ticker_history.index.tz_localize(None)
    ticker_history = ticker_history.resample('D').ffill()

    ticker_history = pd.concat([ticker_history, ticker_history.pct_change(365)], axis=1, keys=[
                               'value', 'annual_value_return'])

    # ticker_history['color'] = ticker_history['annual_value_return'].apply(
    #     lambda x: "#14B3EB" if x > 0 else "#EB4C14")

    return ticker_history


@st.cache_data
def fetch_fx_rate_history_2(asset_currency: str, base_currency: str):
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
