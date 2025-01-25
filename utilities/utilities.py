""" This module contains utility functions for the portfolio app """
from typing import List, TypedDict, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st


class AssetDetails(TypedDict):
    """ TypedDict for asset details sent by the user"""
    symbol: str
    position: float
    Avg_Price: float
    Cost_Basis: float


class ExpandedAssetDetails(AssetDetails):
    """ TypedDict for expanded asset details """
    info: dict
    history: pd.DataFrame


@st.cache_data
def fetch_asset_info_and_history(symbol: str) -> Tuple[dict, pd.Series]:
    """ Fetch the asset info for a given symbol """
    y_finance_ticker = yf.Ticker(symbol)

    ticker_history = y_finance_ticker.history(
        period='max', auto_adjust=True)['Close']
    ticker_history.index = ticker_history.index.tz_localize(None)
    ticker_history = ticker_history.resample('D').ffill()

    ticker_history = pd.concat([ticker_history, ticker_history.pct_change(365)], axis=1, keys=[
                               'value', 'annual_value_return'])

    ticker_history['color'] = ticker_history['annual_value_return'].apply(
        lambda x: "#14B3EB" if x > 0 else "#EB4C14")

    return y_finance_ticker.info, ticker_history


@st.cache_data
def fetch_fx_rate_history(asset_currency: str, base_currency: str) -> pd.Series:
    """ Fetch the fx rate for a given currency pair """

    if asset_currency == base_currency:
        return pd.Series(1, index=pd.date_range(start='1950-01-01', end=pd.Timestamp.today(), freq='D'))

    currency_ticker_name = asset_currency + base_currency + "=X"
    fx_ticker = yf.Ticker(currency_ticker_name)
    fx_history = fx_ticker.history(period='max')['Close']
    fx_history = fx_history.resample('D').ffill()
    fx_history.index = fx_history.index.tz_localize(None)

    return fx_history


def generate_asset_base_value(asset_history: pd.DataFrame, fx_history: pd.Series) -> pd.DataFrame:
    """ Generate the base value for an asset """

    asset_history['fx_history'] = fx_history

    asset_history['base_value'] = asset_history['value'] * \
        asset_history['fx_history']

    asset_history.dropna(subset=['base_value'], inplace=True)

    asset_history['annual_base_return'] = asset_history['base_value'].pct_change(
        365)

    return asset_history


def get_exp_fitted_data(y: List[int]) -> List[int]:
    """ Fit the value data to an exponential curve. """
    x = np.arange(len(y))

    y_log = np.log(y)
    z_exp = np.polyfit(x, y_log, 1)
    p_exp = np.poly1d(z_exp)
    y_exp = np.exp(p_exp(x))
    return y_exp


def append_fitted_data(history_data: pd.DataFrame, selected_period: str, col_to_fit='base_value') -> pd.DataFrame:
    """ Append the fitted data to the history data. """

    period_history_data = history_data.copy().tail(
        int(round(int(selected_period.split(" ")[0])*365.25))+1) if "Max" not in selected_period else history_data.copy()

    period_history_data['fitted'] = get_exp_fitted_data(
        period_history_data[col_to_fit].values)

    days = period_history_data.shape[0]

    cagr = (period_history_data[col_to_fit].iloc[-1] /
            period_history_data[col_to_fit].iloc[0]) ** (1 / (days / 365.25)) - 1
    cagr_fitted = (period_history_data['fitted'].iloc[-1] /
                   period_history_data['fitted'].iloc[0]) ** (1 / (days / 365.25)) - 1

    base_over_under = (period_history_data[col_to_fit].iloc[-1] -
                       period_history_data['fitted'].iloc[-1]) / period_history_data['fitted'].iloc[-1]

    period_history_data['Date'] = period_history_data.index

    return period_history_data, cagr, cagr_fitted, base_over_under


def create_asset_info_df(asset_info: dict) -> pd.DataFrame:
    """ Get the asset info DataFrame """
    keys_to_display = [
        {
            "key": 'longName',
            "label": "Name"
        },
        {
            "key": 'currency',
            "label": "Currency"
        },
        {
            "key": 'fundFamily',
            "label": "Fund Family"
        },
        {
            "key": 'legalType',
            "label": "Type"
        },
        {
            "key": 'exchange',
            "label": "Exchange"
        },
        {
            "key": 'sector',
            "label": "Sector"
        },
        {
            "key": 'industry',
            "label": "Industry"
        },
        {
            "key": 'country',
            "label": "Country"
        },
        {
            "key": 'quoteType',
            "label": "Quote Type"
        }
    ]

    asset_info_table = pd.DataFrame(
        columns=['Label', 'Value'],
        data=[
            [item['label'],
                asset_info[item['key']]] for item in keys_to_display if item['key']
            in asset_info])

    return asset_info_table
