""" This module contains utility functions for the portfolio app """
from typing import List, TypedDict
import pandas as pd
import numpy as np
# import streamlit as st
# import altair as alt
from utilities.app_yfinance import search_yf, yf_ticket_info, yf_ticket_history, get_fx_history


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


def get_quotes_by_symbol(search_input: str):
    """ Search for a symbol """
    search_results = search_yf(search_input)

    if isinstance(search_results, dict):
        quotes = search_results.get('quotes', None)
        if quotes:
            result_quotes_df = pd.DataFrame(
                quotes).set_index('symbol') if \
                search_results is not None and \
                quotes is not None and \
                len(quotes) > 0 \
                else None

            return result_quotes_df


def get_history_options(history_length: int):
    """ Get the history options """

    year_check = [i for i in ([
        10 * j for j in range(int(history_length / 10 / 365.25), 0, -1)] + [5, 3, 1])
        if round(i * 365.25) < history_length
    ]

    days_list = [history_length] + [round(i * 365.25) for i in year_check]

    days_names = [str(round(days/365.25, 1) if
                      (abs(round(days/365.25, 1) - round(days/365.25, 0)) > 0.05) else
                      int(round(days/365.25, 1))) +
                  (" Year" if days < 366 else " Years") +
                  (' (max)' if i == 0 else '')
                  for i, days in enumerate(days_list) if days > 0]

    return dict(zip(days_names, days_list))


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


def append_fitted_data(history_data: pd.DataFrame, selected_period: int, col_to_fit='base_value'):
    """ Append the fitted data to the history data. """

    period_history_data = history_data.copy().tail(selected_period)

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


def create_asset_info_df(asset_info: dict) -> list[tuple[str, str]]:
    """ Get the asset info DataFrame """
    keys_to_display = [
        {
            "key": 'longName',
            "label": "Name"
        },
        {
            "key": 'legalType',
            "label": "Type"
        },
        {
            "key": 'quoteType',
            "label": "Quote Type"
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
            "key": 'longBusinessSummary',
            "label": "Business Summary"
        }
    ]

    return [[item['label'], asset_info[item['key']]]
            for item in keys_to_display if item['key'] in asset_info]


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

    if 'value' in periodic_asset_history_with_fit.columns:
        latest_value = periodic_asset_history_with_fit['value'].iloc[-1]
        oldest_value = periodic_asset_history_with_fit['value'].iloc[0]
        cagr_value = (latest_value / oldest_value) ** (1 / (days / 365.25)) - 1
    else:
        cagr_value = None

    return pd.DataFrame(
        columns=['Label', 'Value'],
        data=[
            ['Sample Years', f"{days / 365.25:.1f}"]] +
        ([['CAGR Base-fx', f"{(cagr-cagr_value):.1%}"],
          ['CAGR Base-inv', f"{(cagr_value):.1%}"]] if cagr_value is not None else []) +
        [['CAGR Base', f"{cagr:.1%}"],
         ['CAGR Fitted', f"{cagr_fitted:.1%}"],
         ['', ''],
         ['Date',
             periodic_asset_history_with_fit['Date'].iloc[-1].strftime('%Y-%m-%d')],
         ['Base Value',
             f"{periodic_asset_history_with_fit[base_column].iloc[-1]:,.2f}"],
         ['Fitted Value',
             f"{periodic_asset_history_with_fit['fitted'].iloc[-1]:,.2f}"],
         ['Base Over/Under', f"{base_over_under:.1%}"]]
    )


# def display_trend_line_chart(periodic_asset_history_with_fit: pd.DataFrame, base_column='base_value'):
#     """ Display the trend line chart """
#     cagr, cagr_fitted, _ = get_trend_stats(
#         periodic_asset_history_with_fit, base_column)

#     y_axis_padding = 0.1 * \
#         (periodic_asset_history_with_fit[base_column].max() -
#          periodic_asset_history_with_fit[base_column].min())

#     y_axis_start = periodic_asset_history_with_fit[[base_column,
#                                                     'fitted']].min().min() - y_axis_padding

#     y_axis_end = periodic_asset_history_with_fit[[base_column,
#                                                   'fitted']].max().max() + y_axis_padding

#     chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
#         x=alt.X('Date:T', title=None),
#         y=alt.Y(f'{base_column}:Q', scale=alt.Scale(
#             domain=[y_axis_start, y_axis_end]), title=f'Base Value {cagr:.1%}'),
#         color=alt.value('#14B3EB')  # Set custom color
#     )

#     fitted_chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
#         x=alt.X('Date:T', title=None),
#         y=alt.Y('fitted:Q', scale=alt.Scale(
#             domain=[y_axis_start, y_axis_end]), title=f'Fitted Value {cagr_fitted:.1%}'),
#         color=alt.value('#EB4C14')  # Set custom color
#     )

#     st.altair_chart(
#         (chart + fitted_chart).properties(height=425), use_container_width=True)


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


def get_fx_history_2(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""
    currency_symbol = target_currency + base_currency + "=X"
    crypto_symbol = target_currency + "-" + base_currency

    currency_rate_history = yf_ticket_history(
        currency_symbol)
    crypto_rate_history = yf_ticket_history(
        crypto_symbol)

    return currency_rate_history if not currency_rate_history.empty else crypto_rate_history, \
        currency_symbol if not currency_rate_history.empty else crypto_symbol


def fetch_fx_rate_history(asset_currency: str, base_currency: str):
    """ Fetch the fx rate for a given currency pair """

    if asset_currency == base_currency:
        return pd.Series(1, index=pd.date_range(start='1950-01-01', end=pd.Timestamp.today(), freq='D'))

    currency_ticker_name = asset_currency + base_currency + "=X"

    fx_history = yf_ticket_history(currency_ticker_name)

    fx_history = fx_history.resample('D').ffill()
    fx_history.index = fx_history.index.tz_localize(None)

    return fx_history


def ticker_yf_history(symbol: str):
    """ Fetch the asset info for a given symbol """
    ticker_history = yf_ticket_history(symbol)

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = ticker_history.index.tz_localize(None)

    ticker_history = ticker_history.resample('D').ffill()

    ticker_history = pd.concat([ticker_history, ticker_history.pct_change(365)], axis=1, keys=[
                               'value', 'annual_value_return'])

    return ticker_history


def localize_and_fill(ticker_history: pd.DataFrame | pd.Series):
    """ Localize and fill the ticker history """
    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = ticker_history.index.tz_localize(None)
    ticker_history = ticker_history.resample('D').ffill()

    return ticker_history


def calculate_financial_metrics(df: pd.DataFrame, value_col: str, fitted_col: str, return_col: str):
    """ Show the stats for the given DataFrame """
    start_date = df.index[0]
    last_date = df.index[-1]
    actual_days_duration = (last_date - df.index[0]).days
    actual_years_duration = actual_days_duration / 365.25
    avg_points_per_year = df.shape[0]/actual_years_duration

    cagr = (df[value_col].iloc[-1] /
            df[value_col].iloc[0]) ** (1 / actual_years_duration) - 1
    cagr_fitted = (df[fitted_col].iloc[-1] /
                   df[fitted_col].iloc[0]) ** (1 / actual_years_duration) - 1

    over_under = (df[value_col].iloc[-1] -
                  df[fitted_col].iloc[-1]) / df[fitted_col].iloc[-1]

    daily_returns_mean = df[return_col].mean()
    daily_return_std = df[return_col].std()

    annualized_returns_to_risk_ratio = (daily_returns_mean / daily_return_std) * \
        np.sqrt(avg_points_per_year)

    annualized_return = (
        1 + daily_returns_mean) ** avg_points_per_year - 1
    annualized_risk = daily_return_std * np.sqrt(avg_points_per_year)

    return {
        'start_date': start_date,
        'last_date': last_date,
        'actual_years_duration': actual_years_duration,
        'cagr': cagr,
        'cagr_fitted': cagr_fitted,
        'over_under': over_under,
        'annualized_return': annualized_return,
        'annualized_risk': annualized_risk,
        'annualized_returns_to_risk_ratio': annualized_returns_to_risk_ratio
    }


def fully_analyze_symbol(symbol: str,  base_currency: str, years: int):
    """ Fetch the asset info for a given symbol """

    ###############################
    # Fetch from Yahoo Finance API

    # Get asset info to get the currency
    ticker_info = yf_ticket_info(symbol)
    if isinstance(ticker_info, str):
        return ticker_info

    asset_currency = ticker_info.get('currency')
    if not asset_currency:
        return f"Asset currency for {symbol} is not defined in the ticker info"

    # Get asset history
    ticker_history = yf_ticket_history(symbol)
    if isinstance(ticker_history, str):
        return ticker_history
    if ticker_history.shape[0] == 0:
        return f"Retrieved ticker history is empty for {symbol}"

    # Get the fx rate history
    fx_history = get_fx_history(base_currency, asset_currency)
    if isinstance(fx_history, str):
        return fx_history
    if fx_history.shape[0] == 0:
        return f"Retrieved fx history is empty for {base_currency} and {asset_currency}"

    ###############################
    # Localize and Filter data window
    ticker_history.index = ticker_history.index.tz_localize(None)
    fx_history.index = fx_history.index.tz_localize(None)

    start_date = ticker_history.index[-1] - pd.DateOffset(years=years)
    ticker_history = ticker_history[ticker_history.index >= start_date]

    ###############################
    # Trade Value
    trade_value_col = 'Trade Value'
    trade_value_return_col = 'Trade Value Return'
    trade_value_fitted_col = 'Trade Value Fitted'

    trade_df = pd.DataFrame(
        ticker_history.values, index=ticker_history.index, columns=[trade_value_col])
    trade_df[trade_value_return_col] = trade_df[trade_value_col].pct_change()
    trade_df[trade_value_fitted_col] = get_exp_fitted_data(
        trade_df[trade_value_col].values)

    # Home value - TODO: if the same then just copy the trade_df
    fx_col = 'FX Rate'
    home_value_col = 'Home Value'
    home_value_return_col = 'Home Value Return'
    home_value_fitted_col = 'Home Value Fitted'

    home_df = pd.concat(
        [trade_df[trade_value_col], fx_history.rename(fx_col)],
        axis=1,
        join='inner')

    home_df[home_value_col] = home_df[trade_value_col] * home_df[fx_col]
    home_df[home_value_return_col] = home_df[home_value_col].pct_change()
    home_df[home_value_fitted_col] = get_exp_fitted_data(
        home_df[home_value_col].values)

    home_df.drop(columns=[trade_value_col], inplace=True)

    trade_metrics = calculate_financial_metrics(trade_df, trade_value_col,
                                                trade_value_fitted_col, trade_value_return_col)

    home_metrics = calculate_financial_metrics(home_df, home_value_col,
                                               home_value_fitted_col, home_value_return_col)

    return trade_df, trade_metrics, home_df, home_metrics
