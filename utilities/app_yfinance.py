""" Yfinance utilities for Streamlit app """
from dataclasses import dataclass
from typing import Union, TypedDict,  Any
import pandas as pd
import yfinance as yf
import streamlit as st


def get_sector_keys():
    """ Get the sector keys """
    try:
        return list(
            yf.const.SECTOR_INDUSTY_MAPPING.keys())  # type: ignore

    except Exception as err:
        print(f"Error retrieving sector keys: {err}")
        return [
            'basic-materials',
            'communication-services',
            'consumer-cyclical',
            'consumer-defensive',
            'energy',
            'financial-services',
            'healthcare',
            'industrials',
            'real-estate',
            'technology',
            'utilities']


YF_SECTOR_KEYS = get_sector_keys()

# ######### #
# yf.Sector #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


@dataclass
class SectorOverview:
    """Type definition for sector overview"""
    companies_count: int
    market_cap: int
    message_board_id: str
    description: str
    industries_count: int
    market_weight: float
    employee_count: int


class SectorData(TypedDict):
    """
    Type definition for successful sector data response.

    top_companies: pd.DataFrame
        - Columns: "name", "rating", "market weight"
        - Index: "symbol"

    top_etfs: dict[str, str]
        - Key: "symbol"
        - Value: "short name"
    """
    key: str
    name: str
    overview: dict  # is SectorOverview
    top_companies: pd.DataFrame
    top_etfs: dict[str, str]


@st.cache_data
def sector_yf(sector_key: str) -> Union[str, SectorData]:
    """ Fetch the sector data """
    try:
        sector = yf.Sector(sector_key)

        name = sector.name
        overview: dict = sector.overview
        top_companies = sector.top_companies
        top_etfs = sector.top_etfs

        if not isinstance(top_companies, pd.DataFrame):
            return f"Error: Top companies is not a DataFrame: {top_companies}"

        if not isinstance(top_etfs, dict):
            return f"Error: Top etfs is not a dictionary: {top_etfs}"

        return SectorData(
            key=sector_key,
            name=name,
            overview=overview,
            top_companies=top_companies,
            top_etfs=top_etfs,
        )

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
        raise ValueError(
            f"Error retrieving ticker details for '{symbol}': {err}") from err


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

    raise ValueError(
        f"Error retrieving historical data for either '{fx_symbol}' or '{crypto_symbol}'")

# ^^^^^^^^^ #
# yf.Ticker #
# ######### #


# ######### #
# yf.Market #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #

class MarketData(TypedDict):
    """Type definition for successful market data response"""
    summary: dict[str, Any]
    first_summary_symbol: str


@st.cache_data
def market_yf(market: str) -> Union[str, MarketData]:
    """ Fetch the market info for a given market """
    try:
        market_data = yf.Market(market)

        data_summary = market_data.summary

        # To get first market symbol
        if not isinstance(data_summary, dict):
            return "Error: Market summary is not a dictionary"

        summary_keys = list(data_summary.keys())

        first_summary = data_summary[summary_keys[0]]

        if not isinstance(first_summary, dict):
            return "Error: Market symbol is not a dictionary"

        if not 'symbol' in first_summary:
            return "Error: Symbol is not in the first summary"

        first_summary_symbol = first_summary['symbol']

        if not isinstance(first_summary_symbol, str):
            return "Error: Market symbol is not a string"

        return {
            'summary': data_summary,
            'first_summary_symbol': first_summary_symbol
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


@st.cache_data
def tickers_yf(symbols: list[str], period='max') -> TickersData:
    """ Fetch the tickers data """
    try:
        tickers = yf.Tickers(symbols)
        history_data = tickers.history(period=period, auto_adjust=True)
        if history_data is None:
            raise ValueError("Error: No history data returned")

        close_prices = history_data["Close"]

        if not isinstance(close_prices, pd.DataFrame):
            raise ValueError("Error: Close prices is not a DataFrame")

        # drop all columns with all NaNs for erroneous symbols
        close_prices = close_prices.dropna(axis=1, how='all')

        # drop all rows with any NaNs to have consistent data
        close_prices = close_prices.dropna(axis=0, how='any')

        if close_prices.empty:
            raise ValueError("Error: No close prices returned")

        if len(close_prices.columns) != len(symbols):
            raise ValueError(
                f"\nError: Not all symbols have data: {[el for el in symbols if el not in close_prices.columns]}\n")

        return {
            "history": close_prices
        }
    except Exception as err:
        error_message = f"Error retrieving tickers details for '{symbols}': {err}"
        raise ValueError(error_message) from err

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
