"""Yfinance utilities for Streamlit app"""

from dataclasses import dataclass
import logging
from typing import TypedDict, Any
import pandas as pd
import yfinance as yf
import streamlit as st

from utilities.result import Result

logger = logging.getLogger(__name__)

YF_CACHE_TTL = "6h"
YF_CACHE_MAX_ENTRIES = 256

YF_RECOVERABLE_ERRORS = (AttributeError, KeyError,
                         TypeError, ValueError, RuntimeError, OSError)

# yf.Ticker.info #


def get_raw_ticker_info(symbol: str) -> Result[dict]:
    """Fetch raw ticker info without Streamlit cache (safe for worker threads)."""
    try:
        return Result(data=yf.Ticker(symbol).info, error=None)
    except Exception as err:
        return Result(data=None, error=f"Error retrieving ticker details for '{symbol}': {err}")


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def yf_ticket_info(symbol: str) -> Result[dict]:
    """Fetch the asset info for a given symbol (cached for main thread use)."""
    return get_raw_ticker_info(symbol)


# yf.Ticker.history #


def get_raw_ticker_history(symbol: str, period: str = "max") -> Result["pd.Series[Any]"]:
    """Fetch the asset history for a given symbol without Streamlit cache."""
    try:
        ticker = yf.Ticker(symbol)
        close = ticker.history(period=period, auto_adjust=True)["Close"]
        return Result(data=close, error=None)
    except Exception as err:
        error_message = f"Error retrieving ticker history for '{symbol}': {err}"
        logger.error(error_message)
        return Result(data=None, error=error_message)


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def yf_ticket_history(symbol: str, period="max") -> Result["pd.Series[Any]"]:
    """Fetch the asset history for a given symbol (cached for main thread use)."""
    return get_raw_ticker_history(symbol, period=period)


def get_sector_keys():
    """Get the sector keys formatted for yf.Sector API"""
    try:
        raw_keys = list(yf.const.SECTOR_INDUSTY_MAPPING.keys())  # type: ignore
        return [k.lower().replace(" ", "-") for k in raw_keys]

    except YF_RECOVERABLE_ERRORS as err:
        logger.warning("Error retrieving sector keys: %s", err)
        return [
            "basic-materials",
            "communication-services",
            "consumer-cyclical",
            "consumer-defensive",
            "energy",
            "financial-services",
            "healthcare",
            "industrials",
            "real-estate",
            "technology",
            "utilities",
        ]


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


def get_raw_sector_data(sector_key: str) -> Result[SectorData]:
    """Fetch sector data without Streamlit cache (safe for worker threads)."""
    try:
        sector = yf.Sector(sector_key)

        name = sector.name
        overview: dict = sector.overview
        top_companies = sector.top_companies
        top_etfs = sector.top_etfs

        if not isinstance(top_companies, pd.DataFrame):
            return Result(data=None, error=f"Error: Top companies is not a DataFrame: {top_companies}")

        if not isinstance(top_etfs, dict):
            return Result(data=None, error=f"Error: Top etfs is not a dictionary: {top_etfs}")

        return Result(
            data=SectorData(
                key=sector_key,
                name=name,
                overview=overview,
                top_companies=top_companies,
                top_etfs=top_etfs,
            ),
            error=None,
        )

    except YF_RECOVERABLE_ERRORS as err:
        error_message = f"Error retrieving sector details for '{sector_key}': {err}"
        return Result(data=None, error=error_message)


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def sector_yf(sector_key: str) -> Result[SectorData]:
    """Fetch the sector data (cached for main thread use)."""
    return get_raw_sector_data(sector_key)


# ######### #
# yf.Search #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #
def get_raw_search(search_input: str) -> Result[dict[str, list]]:
    """Search for a symbol without Streamlit cache (safe for worker threads)."""
    try:
        search_results = yf.Search(search_input)
        return Result(data=search_results.all, error=None)
    except YF_RECOVERABLE_ERRORS as err:
        error_message = f"Error retrieving search results for '{search_input}': {err}"
        logger.error(error_message)
        return Result(data=None, error=error_message)


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def search_yf(search_input: str) -> Result[dict[str, list]]:
    """Search for a symbol (cached for main thread use)."""
    return get_raw_search(search_input)


def get_raw_ticker_currency(symbol: str) -> Result[str]:
    """Fetch the trading currency for a symbol without Streamlit cache."""
    try:
        ticker = yf.Ticker(symbol)

        fast_info = getattr(ticker, "fast_info", None)
        if fast_info is not None:
            currency = fast_info.get("currency")
            if isinstance(currency, str) and currency.strip():
                return Result(data=currency.strip().upper(), error=None)

        info = ticker.info
        currency = info.get("currency") or info.get("financialCurrency")
        if isinstance(currency, str) and currency.strip():
            return Result(data=currency.strip().upper(), error=None)

        return Result(data=None, error=f"Currency not found for '{symbol}'")
    except YF_RECOVERABLE_ERRORS as err:
        return Result(data=None, error=f"Error retrieving currency for '{symbol}': {err}")


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def yf_ticket_currency(symbol: str) -> Result[str]:
    """Fetch the trading currency for a symbol without relying on full .info."""
    return get_raw_ticker_currency(symbol)


def get_raw_fx_history(base_currency: str, target_currency: str) -> Result["pd.Series[Any]"]:
    """Get the historical data of a currency pair without Streamlit cache."""
    if base_currency == target_currency:
        same_currency_series = pd.Series(
            1, index=pd.date_range(start="1980-01-01", end=pd.Timestamp.today(), freq="D"))
        return Result(data=same_currency_series, error=None)

    fx_symbol = target_currency + base_currency + "=X"
    fx_result = get_raw_ticker_history(fx_symbol)

    if fx_result.error is None:
        return fx_result

    crypto_symbol = target_currency + "-" + base_currency

    crypto_result = get_raw_ticker_history(crypto_symbol)

    if crypto_result.error is None:
        return crypto_result

    return Result(
        data=None,
        error=f"Error retrieving historical data for either '{fx_symbol}' or '{crypto_symbol}'",
    )


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def get_fx_history(base_currency, target_currency) -> Result["pd.Series[Any]"]:
    """Get the historical data of a currency pair from Yahoo Finance API."""
    return get_raw_fx_history(base_currency, target_currency)


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


def get_raw_market_data(market: str) -> Result[MarketData]:
    """Fetch market info for a given market without Streamlit cache."""
    try:
        market_data = yf.Market(market)

        data_summary = market_data.summary

        # To get first market symbol
        if not isinstance(data_summary, dict):
            return Result(data=None, error="Error: Market summary is not a dictionary")

        summary_keys = list(data_summary.keys())

        first_summary = data_summary[summary_keys[0]]

        if not isinstance(first_summary, dict):
            return Result(data=None, error="Error: Market symbol is not a dictionary")

        if not "symbol" in first_summary:
            return Result(data=None, error="Error: Symbol is not in the first summary")

        first_summary_symbol = first_summary["symbol"]

        if not isinstance(first_summary_symbol, str):
            return Result(data=None, error="Error: Market symbol is not a string")

        return Result(
            data={"summary": data_summary, "first_summary_symbol": first_summary_symbol}, error=None
        )
    except YF_RECOVERABLE_ERRORS as err:
        error_message = f"Error retrieving market details for '{market}': {err}"
        return Result(data=None, error=error_message)


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def market_yf(market: str) -> Result[MarketData]:
    """Fetch the market info for a given market (cached for main thread use)."""
    return get_raw_market_data(market)


# ^^^^^^^^^ #
# yf.Market #
# ######### #

# ######### #
# yf.Tickers #
# ⌄⌄⌄⌄⌄⌄⌄⌄⌄ #


class TickersData(TypedDict):
    """Type definition for successful tickers data response"""

    history: pd.DataFrame


def get_raw_tickers_data(symbols: list[str], period: str = "max") -> Result[TickersData]:
    """Fetch tickers data without Streamlit cache."""
    try:
        tickers = yf.Tickers(symbols)
        history_data = tickers.history(period=period, auto_adjust=True)
        if history_data is None:
            return Result(data=None, error="Error: No history data returned")

        close_prices = history_data["Close"]

        if not isinstance(close_prices, pd.DataFrame):
            return Result(data=None, error="Error: Close prices is not a DataFrame")

        # drop all columns with all NaNs for erroneous symbols
        close_prices = close_prices.dropna(axis=1, how="all")

        # drop all rows with any NaNs to have consistent data
        close_prices = close_prices.dropna(axis=0, how="any")

        if close_prices.empty:
            return Result(data=None, error="Error: No close prices returned")

        if len(close_prices.columns) != len(symbols):
            missing = [el for el in symbols if el not in close_prices.columns]
            return Result(data=None, error=f"Error: Not all symbols have data: {missing}")

        return Result(data={"history": close_prices}, error=None)
    except Exception as err:
        error_message = f"Error retrieving tickers details for '{symbols}': {err}"
        return Result(data=None, error=error_message)


@st.cache_data(ttl=YF_CACHE_TTL, max_entries=YF_CACHE_MAX_ENTRIES)
def tickers_yf(symbols: list[str], period="max") -> Result[TickersData]:
    """Fetch the tickers data"""
    return get_raw_tickers_data(symbols, period=period)


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
