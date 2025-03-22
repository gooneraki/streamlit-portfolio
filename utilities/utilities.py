""" This module contains utility functions for the portfolio app """
from typing import List, TypedDict
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


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
def fetch_asset_info(symbol: str):
    """ Fetch the asset info for a given symbol """
    y_finance_ticker = yf.Ticker(symbol)
    return y_finance_ticker.info


@st.cache_data
def fetch_asset_history(symbol: str):
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


def generate_asset_base_value(asset_history: pd.DataFrame, fx_history: pd.Series):
    """ Generate the base value for an asset """

    asset_history['fx_history'] = fx_history

    asset_history['base_value'] = asset_history['value'] * \
        asset_history['fx_history']

    asset_history.dropna(subset=['base_value'], inplace=True)

    asset_history['annual_base_return'] = asset_history['base_value'].pct_change(
        365)

    return asset_history


def get_exp_fitted_data(y: List[int]):
    """ Fit the value data to an exponential curve. """
    if len(y) < 2:
        return y

    x = np.arange(len(y))

    y_log = np.log(y)
    z_exp = np.polyfit(x, y_log, 1)
    p_exp = np.poly1d(z_exp)
    y_exp = np.exp(p_exp(x))
    return y_exp


def append_fitted_data(history_data: pd.DataFrame, selected_period: str, col_to_fit='base_value') -> pd.DataFrame:
    """ Append the fitted data to the history data. """
    tail_n = int(round(int(selected_period.split(" ")[
                 0])*365.25)) + 1 if "Max" not in selected_period else 0

    period_history_data = history_data.copy().tail(
        tail_n) if "Max" not in selected_period else history_data.copy()

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


def create_asset_info_df(asset_info: dict):
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
        columns=['Field', 'Value'],
        data=[
            [item['label'], asset_info[item['key']]]
            for item in keys_to_display if item['key'] in asset_info])

    return asset_info_table


def get_trend_stats(periodic_asset_history_with_fit: pd.DataFrame, base_column='base_value'):
    """ Get the trend stats """
    days = periodic_asset_history_with_fit.shape[0]
    latest_base_value = periodic_asset_history_with_fit[base_column].iloc[-1]
    latest_fitted_value = periodic_asset_history_with_fit['fitted'].iloc[-1]
    oldest_base_value = periodic_asset_history_with_fit[base_column].iloc[0]
    oldest_fitted_value = periodic_asset_history_with_fit['fitted'].iloc[0]

    cagr = (latest_base_value / oldest_base_value) ** (1 / (days / 365.25)) - 1
    cagr_fitted = (latest_fitted_value /
                   oldest_fitted_value) ** (1 / (days / 365.25)) - 1

    base_over_under = latest_base_value / \
        latest_fitted_value - 1

    return cagr, cagr_fitted, base_over_under


def get_trend_info(periodic_asset_history_with_fit: pd.DataFrame, base_column='base_value') -> pd.DataFrame:
    """ Get the trend info DataFrame """

    days = periodic_asset_history_with_fit.shape[0]
    latest_base_value = periodic_asset_history_with_fit[base_column].iloc[-1]
    latest_fitted_value = periodic_asset_history_with_fit['fitted'].iloc[-1]
    oldest_base_value = periodic_asset_history_with_fit[base_column].iloc[0]
    oldest_fitted_value = periodic_asset_history_with_fit['fitted'].iloc[0]

    cagr = (latest_base_value / oldest_base_value) ** (1 / (days / 365.25)) - 1
    cagr_fitted = (latest_fitted_value /
                   oldest_fitted_value) ** (1 / (days / 365.25)) - 1

    base_over_under = latest_base_value / \
        latest_fitted_value - 1

    return pd.DataFrame(
        columns=['Label', 'Value'],
        data=[
            ['Sample Years', f"{days / 365.25:.1f}"],
            ['CAGR Base', f"{cagr:.1%}"],
            ['CAGR Fitted', f"{cagr_fitted:.1%}"],
            ['', ''],
            ['Date',
                periodic_asset_history_with_fit['Date'].iloc[-1].strftime('%Y-%m-%d')],
            ['Base Value', f"{
                periodic_asset_history_with_fit[base_column].iloc[-1]:,.2f}"],
            ['Fitted Value', f"{
                periodic_asset_history_with_fit['fitted'].iloc[-1]:,.2f}"],
            ['Base Over/Under', f"{base_over_under:.1%}"]
        ]
    )


def display_trend_line_chart(periodic_asset_history_with_fit: pd.DataFrame, base_column='base_value'):
    """ Display the trend line chart """
    cagr, cagr_fitted, _ = get_trend_stats(
        periodic_asset_history_with_fit, base_column)

    y_axis_padding = 0.1 * \
        (periodic_asset_history_with_fit[base_column].max() -
         periodic_asset_history_with_fit[base_column].min())

    y_axis_start = periodic_asset_history_with_fit[[base_column,
                                                    'fitted']].min().min() - y_axis_padding

    y_axis_end = periodic_asset_history_with_fit[[base_column,
                                                  'fitted']].max().max() + y_axis_padding

    chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
        x=alt.X('Date:T', title=None),
        y=alt.Y(f'{base_column}:Q', scale=alt.Scale(
            domain=[y_axis_start, y_axis_end]), title=f'Base Value {cagr:.1%}'),
        color=alt.value('#14B3EB')  # Set custom color
    )

    fitted_chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
        x=alt.X('Date:T', title=None),
        y=alt.Y('fitted:Q', scale=alt.Scale(
            domain=[y_axis_start, y_axis_end]), title=f'Fitted Value {cagr_fitted:.1%}'),
        color=alt.value('#EB4C14')  # Set custom color
    )

    st.altair_chart(
        (chart + fitted_chart).properties(height=425), use_container_width=True)


def get_annual_returns_trend_info(periodic_asset_history_with_fit: pd.DataFrame):
    """ Get the annual trend info DataFrame """

    clean_df = periodic_asset_history_with_fit.dropna()

    if clean_df is None or clean_df.shape[0] == 0:
        return None, None

    mean = clean_df['annual_base_return'].mean()

    return pd.DataFrame(
        columns=['Label', 'Value'],
        data=[
            ['Sample Years', f"{
                clean_df.shape[0] / 365.25:.1f}"],
            ['Mean Annual Return', f"{mean:.1%}"],
            ['', ''],
            ['Date',
             clean_df['Date'].iloc[-1].strftime('%Y-%m-%d')],
            ['Current Annual Return', f"{
                clean_df['annual_base_return'].iloc[-1]:,.1%}"],
        ]
    ), mean
