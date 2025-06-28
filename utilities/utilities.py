""" This module contains utility functions for the portfolio app """
from typing import List, TypedDict, Union, Any
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
# import streamlit as st
# import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utilities.app_yfinance import YF_SECTOR_KEYS, \
    search_yf, tickers_yf, yf_ticket_info, yf_ticket_history, get_fx_history, sector_yf


class FormatType(Enum):
    """ Format type enum """
    FLOAT_1 = ".1f"
    FLOAT_2 = ".2f"
    PERCENTAGE_1 = ".1%"
    INTEGER = "d"
    DATE = "%Y-%m-%d"


@dataclass
class Metric:
    """ Metric class """
    key: str
    label: str
    fmt: FormatType


metrics = {
    'start_date': Metric('start_date', 'Start Date', FormatType.DATE),
    'last_date': Metric('last_date', 'Last Date', FormatType.DATE),
    'actual_years_duration': Metric('actual_years_duration', 'Years', FormatType.FLOAT_1),
    'cagr': Metric('cagr', 'CAGR', FormatType.PERCENTAGE_1),
    'cagr_fitted': Metric('cagr_fitted', 'CAGR Fitted', FormatType.PERCENTAGE_1),
    'over_under': Metric('over_under', 'Over/Under', FormatType.PERCENTAGE_1),
    'annualized_return': Metric('annualized_return', 'Annualized Return', FormatType.PERCENTAGE_1),
    'annualized_risk': Metric('annualized_risk', 'Annualized Risk', FormatType.PERCENTAGE_1),
    'annualized_returns_to_risk_ratio': Metric('annualized_returns_to_risk_ratio',
                                               'Return/Risk Ratio', FormatType.FLOAT_2)

}


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


def get_exp_fitted_data(y: Union[List[float], np.ndarray, pd.Series, Any]):
    """ Fit the value data to an exponential curve. """
    if len(y) < 2:
        return y

    # Convert to numpy array if needed
    if isinstance(y, (list, pd.Series)):
        y = np.array(y)

    x = np.arange(len(y))

    y_log = np.log(y)
    z_exp = np.polyfit(x, y_log, 1)
    p_exp = np.poly1d(z_exp)
    y_exp = np.exp(p_exp(x))

    return y_exp


def get_rolling_exp_fit(series: pd.Series, min_points: int = 2):
    """Calculate exponential fit for each point using all historical data up to that point"""
    # Need at least 2 points for a fit
    if min_points < 2:
        min_points = 2

    result = pd.Series(index=series.index, dtype=float)

    for i in range(len(series)):
        if i < min_points:
            result.iloc[i] = series.iloc[i]
            continue

        # Get all data up to current point
        historical_data = series.iloc[:i+1]
        x = np.arange(len(historical_data))
        y = historical_data.values

        # Fit exponential
        y_log = np.log(y)
        z_exp = np.polyfit(x, y_log, 1)
        p_exp = np.poly1d(z_exp)

        # Get the fitted value for the current point
        result.iloc[i] = np.exp(p_exp(len(historical_data)-1))

    return result


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


def create_asset_info_df(asset_info: dict) -> List[List[str]]:
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

    return [[item['label'], str(asset_info.get(item['key'], ''))]
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
        data=[
            ['Sample Years', f"{days / 365.25:.1f}"],
            ['CAGR Base-fx', f"{(cagr-cagr_value):.1%}"],
            ['CAGR Base-inv',
                f"{(cagr_value):.1%}"] if cagr_value is not None else [],
            ['CAGR Base', f"{cagr:.1%}"],
            ['CAGR Fitted', f"{cagr_fitted:.1%}"],
            ['', ''],
            ['Date',
                periodic_asset_history_with_fit['Date'].iloc[-1].strftime('%Y-%m-%d')],
            ['Base Value',
                f"{periodic_asset_history_with_fit[base_column].iloc[-1]:,.2f}"],
            ['Fitted Value',
                f"{periodic_asset_history_with_fit['fitted'].iloc[-1]:,.2f}"],
            ['Base Over/Under', f"{base_over_under:.1%}"]],
        columns=pd.Index(['Label', 'Value'])
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

    data_rows = [
        ['Sample Years', f"{clean_df.shape[0] / 365.25:.1f}"],
        ['Mean Annual Return', f"{mean:.1%}"],
        ['', ''],
        ['Date', clean_df['Date'].iloc[-1].strftime('%Y-%m-%d')],
        ['Current Annual Return',
            f"{clean_df['annual_base_return'].iloc[-1]:,.1%}"],
    ]

    return pd.DataFrame(data_rows, columns=pd.Index(['Label', 'Value'])), mean


def get_fx_history_2(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""
    currency_symbol = target_currency + base_currency + "=X"
    crypto_symbol = target_currency + "-" + base_currency

    currency_rate_history = yf_ticket_history(currency_symbol)
    crypto_rate_history = yf_ticket_history(crypto_symbol)

    if isinstance(currency_rate_history, pd.Series) and not currency_rate_history.empty:
        return currency_rate_history, currency_symbol
    elif isinstance(crypto_rate_history, pd.Series) and not crypto_rate_history.empty:
        return crypto_rate_history, crypto_symbol
    else:
        return None, None


def fetch_fx_rate_history(asset_currency: str, base_currency: str):
    """ Fetch the fx rate for a given currency pair """

    if asset_currency == base_currency:
        return pd.Series(1, index=pd.date_range(start='1950-01-01', end=pd.Timestamp.today(), freq='D'))

    currency_ticker_name = asset_currency + base_currency + "=X"

    fx_history = yf_ticket_history(currency_ticker_name)

    if isinstance(fx_history, str):
        return fx_history

    fx_history = fx_history.resample('D').ffill()
    fx_history.index = pd.DatetimeIndex(fx_history.index).tz_localize(None)

    return fx_history


def ticker_yf_history(symbol: str):
    """ Fetch the asset info for a given symbol """
    ticker_history = yf_ticket_history(symbol)

    if isinstance(ticker_history, str):
        return None

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)

    ticker_history = ticker_history.resample('D').ffill()

    ticker_history = pd.concat([ticker_history, ticker_history.pct_change(365)], axis=1, keys=[
                               'value', 'annual_value_return'])

    return ticker_history


def localize_and_fill(ticker_history: Union[pd.DataFrame, pd.Series]):
    """ Localize and fill the ticker history """
    if isinstance(ticker_history, str):
        return None

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)
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
        metrics['start_date'].key: start_date,
        metrics['last_date'].key: last_date,
        metrics['actual_years_duration'].key: actual_years_duration,
        metrics['cagr'].key: cagr,
        metrics['cagr_fitted'].key: cagr_fitted,
        metrics['over_under'].key: over_under,
        metrics['annualized_return'].key: annualized_return,
        metrics['annualized_risk'].key: annualized_risk,
        metrics['annualized_returns_to_risk_ratio'].key: annualized_returns_to_risk_ratio,
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
    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)
    fx_history.index = pd.DatetimeIndex(fx_history.index).tz_localize(None)

    start_date = ticker_history.index[-1] - pd.DateOffset(years=years)
    ticker_history = ticker_history[ticker_history.index >= start_date]

    ###############################
    # Trade Value
    trade_value_col = 'Trade Value'
    trade_value_return_col = 'Trade Value Return'
    trade_value_fitted_col = 'Trade Value Fitted'

    # Ensure ticker_history is a pandas Series
    if isinstance(ticker_history, pd.Series):
        ticker_series = ticker_history
    else:
        ticker_series = pd.Series(ticker_history)

    trade_df = pd.DataFrame(
        ticker_series.values, index=ticker_series.index, columns=pd.Index([trade_value_col]))
    trade_df[trade_value_return_col] = trade_df[trade_value_col].pct_change()
    trade_df[trade_value_fitted_col] = get_exp_fitted_data(
        trade_df[trade_value_col].values)

    # Home value - TODO: if the same then just copy the trade_df
    fx_col = 'FX Rate'
    home_value_col = 'Home Value'
    home_value_return_col = 'Home Value Return'
    home_value_fitted_col = 'Home Value Fitted'

    # Rename the fx_history series properly
    fx_history_renamed = fx_history.copy()
    fx_history_renamed.name = fx_col

    home_df = pd.concat(
        [trade_df[trade_value_col], fx_history_renamed],
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


def retrieve_sector_data(home_currency: str, data_years: int):
    """Retrieve the sector data"""
    sectors_data = list(sorted(
        [sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS],
        key=lambda x: x["overview"]["market_weight"] if isinstance(
            x, dict) and "overview" in x and "market_weight" in x["overview"] else -99,
        reverse=True,
    ))

    # Process sectors and add etf_metrics
    processed_sectors = []
    for sector in sectors_data:
        if not isinstance(sector, dict):
            processed_sectors.append(sector)
            continue

        top_etfs = sector.get("top_etfs")
        if (
            top_etfs is not None
            and isinstance(top_etfs, dict)
            and len(list(top_etfs.keys())) > 0
        ):

            first_etf = list(top_etfs.keys())[0]
            info = yf_ticket_info(first_etf)
            _, etf_trade_metrics, _, etf_home_metrics = fully_analyze_symbol(
                first_etf, home_currency, data_years
            )

            etf_metrics = {
                "top_etf": first_etf,
                "etf_info": info,
                "etf_trade_metrics": etf_trade_metrics,
                "etf_home_metrics": etf_home_metrics,
            }

            # Create a new sector dict with etf_metrics
            sector_with_etf = dict(sector)
            sector_with_etf["etf_metrics"] = etf_metrics
            processed_sectors.append(sector_with_etf)
        else:
            processed_sectors.append(sector)

    data = [
        {
            "Sector":  sector["name"],
            "Market Weight": sector["overview"]["market_weight"],
            "ETF": sector["etf_metrics"]["top_etf"],
            # "ETF Summary":   sector["etf_metrics"]["etf_info"]["longBusinessSummary"],
            "Trade Over/Under":    sector["etf_metrics"]["etf_trade_metrics"]["over_under"],
            "Trade CAGR":   sector["etf_metrics"]["etf_trade_metrics"]["cagr"],
            "Trade CAGR Fitted":  sector["etf_metrics"]["etf_trade_metrics"]["cagr_fitted"],
            "Trade Annualized Return":   sector["etf_metrics"]["etf_trade_metrics"]["annualized_return"],
            "Trade Annualized Risk":     sector["etf_metrics"]["etf_trade_metrics"]["annualized_risk"],
            "Trade Return/Risk Ratio":    sector["etf_metrics"]["etf_trade_metrics"][
                "annualized_returns_to_risk_ratio"
            ],
            "Reference Date":   sector["etf_metrics"]["etf_home_metrics"]["last_date"],
            "Sample Years":  sector["etf_metrics"]["etf_home_metrics"]["actual_years_duration"],
        }
        for sector in processed_sectors if isinstance(sector, dict) and "etf_metrics" in sector
    ]

    total_row = {
        "Sector": "Total",
        "Market Weight": sum(sector["Market Weight"] for sector in data),
        "Trade CAGR": sum(sector["Trade CAGR"]*sector["Market Weight"] for sector in data),
    }

    return sectors_data, pd.DataFrame(data + [total_row])


def fetch_multiple_sectors_data():
    """ Fetch the sector data for the given keys """
    sector_data_raw = [sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS]

    sector_errors = [data for data in sector_data_raw if isinstance(data, str)]
    sector_data = [
        data for data in sector_data_raw if isinstance(data, dict)]

    sector_data = sorted(
        sector_data,
        key=lambda x: x["overview"]["market_weight"] if isinstance(
            x, dict) and "overview" in x and "market_weight" in x["overview"] else -99,
        reverse=True,
    )

    # check sum of sector weights
    sum_of_weights = sum(sector["overview"]["market_weight"]
                         for sector in sector_data)
    if round(sum_of_weights, 2) != 1:
        print(f"\nError: Sum of sector weights is not 1: {sum_of_weights}\n")

    return sector_data, sector_errors


def get_period_info(p_history: pd.DataFrame):
    """ Get the period info for the given history """
    first_date = p_history.index[0]
    last_date = p_history.index[-1]

    if not isinstance(first_date, pd.Timestamp):
        raise ValueError("First date is not a Timestamp")
    if not isinstance(last_date, pd.Timestamp):
        raise ValueError("Last date is not a Timestamp")

    number_of_points = p_history.shape[0]
    number_of_days = (last_date - first_date).days + 1
    number_of_years = number_of_days / 365.25
    points_per_year = number_of_points / number_of_years
    points_per_month = points_per_year / 12

    return number_of_points, number_of_days, number_of_years, points_per_year, points_per_month


def get_first_last_values(p_history: pd.DataFrame, p_fitted_history: pd.DataFrame):
    """ Get the first and last values from the history and fitted history """

    first_value: pd.Series = p_history.iloc[0]
    last_value: pd.Series = p_history.iloc[-1]

    first_fitted_value: pd.Series = p_fitted_history.iloc[0]
    last_fitted_value: pd.Series = p_fitted_history.iloc[-1]

    if not isinstance(first_value, pd.Series):
        raise ValueError("First value is not a Series")

    if not isinstance(last_value, pd.Series):
        raise ValueError("Last value is not a Series")

    if not isinstance(first_fitted_value, pd.Series):
        raise ValueError("First fitted value is not a Series")

    if not isinstance(last_fitted_value, pd.Series):
        raise ValueError("Last fitted value is not a Series")

    return first_value, last_value, first_fitted_value, last_fitted_value


def get_history_exp_fit(history: pd.DataFrame):
    """ Get the tickers data extended """

    fitted_history = history.apply(get_exp_fitted_data)
    trend_deviation = history/fitted_history - 1

    if not isinstance(fitted_history, pd.DataFrame):
        raise ValueError("Fitted history is not a DataFrame")

    if not isinstance(trend_deviation, pd.DataFrame):
        raise ValueError("CAGR error is not a DataFrame")

    first_value, last_value, first_fitted_value, last_fitted_value = get_first_last_values(
        history, fitted_history)

    _, _, number_of_years, _, _ = get_period_info(history)

    cagr = (last_value / first_value) ** (1 / number_of_years) - 1
    cagr_fitted = (
        last_fitted_value / first_fitted_value) ** (1 / number_of_years) - 1
    over_under = (
        last_value - last_fitted_value) / last_fitted_value

    if not isinstance(cagr, pd.Series):
        raise ValueError("Error: CAGR is not a Series")

    if not isinstance(cagr_fitted, pd.Series):
        raise ValueError("Error: CAGR fitted is not a Series")

    if not isinstance(over_under, pd.Series):
        raise ValueError("Error: Over/Under is not a Series")

    trend_deviation_rmse = np.sqrt((trend_deviation ** 2).mean())

    if not isinstance(trend_deviation_rmse, pd.Series):
        raise ValueError("Error: CAGR error RMSE is not a Series")

    trend_deviation_z_score = (trend_deviation - trend_deviation.mean()) / \
        trend_deviation.std()

    return fitted_history, trend_deviation, cagr, cagr_fitted, over_under, \
        trend_deviation_rmse, trend_deviation_z_score


def get_log_returns(p_history: pd.DataFrame):
    """ Get the log returns for the given history """

    first_date = p_history.index[0]
    last_date = p_history.index[-1]

    if not isinstance(first_date, pd.Timestamp):
        raise ValueError("First date is not a Timestamp")
    if not isinstance(last_date, pd.Timestamp):
        raise ValueError("Last date is not a Timestamp")

    number_of_points = p_history.shape[0]
    number_of_days = (last_date - first_date).days + 1
    number_of_years = number_of_days / 365.25
    points_per_year = number_of_points / number_of_years
    points_per_month = points_per_year / 12

    daily_log_returns: pd.DataFrame = np.log(
        p_history / p_history.shift(1)).dropna()
    monthly_log_returns: pd.DataFrame = np.log(
        p_history / p_history.shift(round(points_per_month))).dropna()
    yearly_log_returns: pd.DataFrame = np.log(
        p_history / p_history.shift(round(points_per_year))).dropna()
    cumulative_log_returns: pd.DataFrame = daily_log_returns.cumsum()

    if not isinstance(cumulative_log_returns, pd.DataFrame):
        raise ValueError("Error: Cumulative log returns is not a DataFrame")

    mean_log_daily_returns = daily_log_returns.mean()
    std_log_daily_returns = daily_log_returns.std()

    mean_log_monthly_returns = monthly_log_returns.mean()
    std_log_monthly_returns = monthly_log_returns.std()

    mean_log_yearly_returns = yearly_log_returns.mean()
    std_log_yearly_returns = yearly_log_returns.std()

    if not isinstance(mean_log_daily_returns, pd.Series):
        raise ValueError("Error: Mean log returns is not a Series")
    if not isinstance(std_log_daily_returns, pd.Series):
        raise ValueError("Error: Std log returns is not a Series")
    if not isinstance(mean_log_monthly_returns, pd.Series):
        raise ValueError("Error: Mean log returns monthly is not a Series")
    if not isinstance(std_log_monthly_returns, pd.Series):
        raise ValueError("Error: Std log returns monthly is not a Series")
    if not isinstance(mean_log_yearly_returns, pd.Series):
        raise ValueError("Error: Mean log returns yearly is not a Series")
    if not isinstance(std_log_yearly_returns, pd.Series):
        raise ValueError("Error: Std log returns yearly is not a Series")

    return cumulative_log_returns, daily_log_returns, monthly_log_returns, yearly_log_returns, \
        mean_log_daily_returns, std_log_daily_returns, mean_log_monthly_returns, \
        std_log_monthly_returns, mean_log_yearly_returns, std_log_yearly_returns


class MultiAsset:
    """ Multi-asset analysis class """

    def __init__(
            self,
            symbols: list[str],
            period: str = 'max',
            debug: bool = False,
            weights: Union[list[float], None] = None):
        """ Initialize MultiAsset with symbols and calculate metrics """
        self.symbols = symbols
        self.period = period
        self.debug = debug

        # Fetch data from Yahoo Finance
        tickers_data = tickers_yf(symbols, period)
        if isinstance(tickers_data, str):
            raise ValueError(tickers_data)

        history = tickers_data['history']
        if not isinstance(history, pd.DataFrame):
            raise ValueError("History is not a DataFrame")

        # Set default weights if not provided
        if weights is None:
            weights = [1/len(history.columns)] * len(history.columns)

        if len(weights) != len(history.columns):
            raise ValueError(
                "Error: Sector weights length does not match symbols length")

        # Make sure the weights sum to 1
        weights_sum = sum(weights)
        self.weights = [
            w / weights_sum for w in weights] if weights_sum > 0 else weights

        # Create weighted total
        history['TOTAL'] = history.mul(self.weights, axis=1).sum(axis=1)

        # Calculate fitted data and metrics
        fitted_history, trend_deviation, \
            cagr, cagr_fitted, over_under, \
            trend_deviation_rmse, trend_deviation_z_score = get_history_exp_fit(
                history)

        # Calculate log returns
        cumulative_log_returns, daily_log_returns, monthly_log_returns, yearly_log_returns, \
            mean_log_daily_returns, std_log_daily_returns, mean_log_monthly_returns, \
            std_log_monthly_returns, mean_log_yearly_returns, std_log_yearly_returns = get_log_returns(
                history)

        # Debug output
        if self.debug:
            self._print_debug_info(history, fitted_history, trend_deviation, daily_log_returns,
                                   cumulative_log_returns, monthly_log_returns, yearly_log_returns,
                                   cagr, cagr_fitted, over_under, trend_deviation_rmse,
                                   mean_log_daily_returns, std_log_daily_returns,
                                   mean_log_monthly_returns, std_log_monthly_returns,
                                   mean_log_yearly_returns, std_log_yearly_returns)

        # Create symbol metrics DataFrame
        weights_series = pd.Series(
            self.weights + [1], index=history.columns)
        self.symbol_metrics = pd.concat(
            [cagr, cagr_fitted, over_under, trend_deviation_rmse, weights_series,
             mean_log_daily_returns, std_log_daily_returns,
             mean_log_monthly_returns, std_log_monthly_returns,
             mean_log_yearly_returns, std_log_yearly_returns],
            axis=1,
            keys=['cagr', 'cagr_fitted', 'over_under', 'trend_deviation_rmse', 'weights',
                  'mean_log_daily_returns', 'std_log_daily_returns',
                  'mean_log_monthly_returns', 'std_log_monthly_returns',
                  'mean_log_yearly_returns', 'std_log_yearly_returns'])

        # Create timeseries data DataFrame
        self.timeseries_data = pd.concat(
            objs=[history, fitted_history, trend_deviation, trend_deviation_z_score,
                  daily_log_returns, cumulative_log_returns,
                  monthly_log_returns, yearly_log_returns],
            axis=1,
            keys=['history', 'fitted_history', 'trend_deviation', 'trend_deviation_z_score',
                  'daily_log_returns', 'cumulative_log_returns',
                  'monthly_log_returns', 'yearly_log_returns'])

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Return symbol metrics and timeseries data """
        return self.symbol_metrics, self.timeseries_data

    def get_period_info(self) -> dict[str, Any]:
        """ Extract and return period information from timeseries data with type checking """

        # Extract dates with type checking
        first_date = self.timeseries_data.index.min()
        last_date = self.timeseries_data.index.max()

        if not isinstance(first_date, pd.Timestamp):
            raise ValueError("First date is not a Timestamp")
        if not isinstance(last_date, pd.Timestamp):
            raise ValueError("Last date is not a Timestamp")

        # Calculate period metrics
        total_days = (last_date - first_date).days + 1
        sample_years = total_days / 365.25
        data_points = len(self.timeseries_data)
        frequency = data_points / sample_years if sample_years > 0 else 0

        return {
            'first_date': first_date,
            'last_date': last_date,
            'total_days': total_days,
            'sample_years': sample_years,
            'data_points': data_points,
            'frequency': frequency
        }

    def create_asset_analysis_chart(self, selected_asset: str) -> go.Figure:
        """ Create dual-axis chart for selected asset with history, fitted history, and z-score """
        # Validate asset exists
        available_assets = self.timeseries_data.columns.get_level_values(
            1).unique().tolist()
        if selected_asset not in available_assets:
            raise ValueError(
                f"Asset '{selected_asset}' not found. Available assets: {available_assets}")

        # Prepare data for selected asset
        history_data = self.timeseries_data[(
            'history', selected_asset)].dropna()
        fitted_data = self.timeseries_data[(
            'fitted_history', selected_asset)].dropna()
        z_score_data = self.timeseries_data[(
            'trend_deviation_z_score', selected_asset)].dropna()

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add history line (primary axis)
        fig.add_trace(
            go.Scatter(
                x=history_data.index,
                y=history_data.values,
                mode='lines',
                name='History',
                line=dict(color='#1f77b4', width=2)
            ),
            secondary_y=False,
        )

        # Add fitted history line (primary axis)
        fig.add_trace(
            go.Scatter(
                x=fitted_data.index,
                y=fitted_data.values,
                mode='lines',
                name='Fitted History',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ),
            secondary_y=False,
        )

        # Add z-score line (secondary axis)
        fig.add_trace(
            go.Scatter(
                x=z_score_data.index,
                y=z_score_data.values,
                mode='lines',
                name='Trend Deviation Z-Score',
                line=dict(color='#d62728', width=1.5),
                opacity=0.7
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title=f"Asset Analysis: {selected_asset}",
            xaxis_title="Date",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1)
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text="Z-Score", secondary_y=True)

        # Add horizontal line at z-score = 0
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      opacity=0.5, secondary_y=True)

        return fig

    def _print_debug_info(self, history, fitted_history, trend_deviation, daily_log_returns,
                          cumulative_log_returns, monthly_log_returns, yearly_log_returns,
                          cagr, cagr_fitted, over_under, trend_deviation_rmse,
                          mean_log_daily_returns, std_log_daily_returns,
                          mean_log_monthly_returns, std_log_monthly_returns,
                          mean_log_yearly_returns, std_log_yearly_returns):
        """ Print debug information """
        print(f"History: {history.head()}")
        print(f"Fitted history: {fitted_history.head()}")
        print(f"CAGR error: {trend_deviation.head()}")
        print(f"Daily log returns: {daily_log_returns.head()}")
        print(f"Cumulative log returns: {cumulative_log_returns.head()}")
        print(f"Monthly log returns: {monthly_log_returns.head()}")
        print(f"Yearly log returns: {yearly_log_returns.head()}")
        print(f"CAGR: {cagr.head()}")
        print(f"CAGR fitted: {cagr_fitted.head()}")
        print(f"Over/Under: {over_under.head()}")
        print(f"CAGR error RMSE: {trend_deviation_rmse.head()}")
        print(f"Mean log daily returns: {mean_log_daily_returns.head()}")
        print(f"Std log daily returns: {std_log_daily_returns.head()}")
        print(f"Mean log monthly returns: {mean_log_monthly_returns.head()}")
        print(f"Std log monthly returns: {std_log_monthly_returns.head()}")
        print(f"Mean log yearly returns: {mean_log_yearly_returns.head()}")
        print(f"Std log yearly returns: {std_log_yearly_returns.head()}")
