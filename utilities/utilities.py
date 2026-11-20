"""This module contains utility functions for the portfolio app"""

import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utilities.app_yfinance import (
    YF_SECTOR_KEYS,
    get_fx_history,
    search_yf,
    sector_yf,
    tickers_yf,
    yf_ticket_history,
    yf_ticket_info,
)
from utilities.result import Result

logger = logging.getLogger(__name__)


class FormatType(Enum):
    """Format type enum"""

    FLOAT_1 = ".1f"
    FLOAT_2 = ".2f"
    PERCENTAGE_1 = ".1%"
    INTEGER = "d"
    DATE = "%Y-%m-%d"


@dataclass
class Metric:
    """Metric class"""

    key: str
    label: str
    fmt: FormatType


metrics = {
    "start_date": Metric("start_date", "Start Date", FormatType.DATE),
    "last_date": Metric("last_date", "Last Date", FormatType.DATE),
    "actual_years_duration": Metric("actual_years_duration", "Years", FormatType.FLOAT_1),
    "cagr": Metric("cagr", "CAGR", FormatType.PERCENTAGE_1),
    "cagr_fitted": Metric("cagr_fitted", "CAGR Fitted", FormatType.PERCENTAGE_1),
    "over_under": Metric("over_under", "Over/Under", FormatType.PERCENTAGE_1),
    "annualized_return": Metric("annualized_return", "Annualized Return", FormatType.PERCENTAGE_1),
    "annualized_risk": Metric("annualized_risk", "Annualized Risk", FormatType.PERCENTAGE_1),
    "annualized_returns_to_risk_ratio": Metric(
        "annualized_returns_to_risk_ratio", "Return/Risk Ratio", FormatType.FLOAT_2
    ),
}


class AssetDetails(TypedDict):
    """TypedDict for asset details sent by the user"""

    symbol: str
    position: float
    Avg_Price: float
    Cost_Basis: float


class ExpandedAssetDetails(AssetDetails):
    """TypedDict for expanded asset details"""

    info: dict
    history: pd.DataFrame


def _calculate_years_in_existence(ts_seconds: Any) -> float | None:
    """Calculate years in existence from epoch timestamp in seconds."""
    if ts_seconds is None:
        return None
    try:
        start_date = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + datetime.timedelta(
            seconds=float(ts_seconds)
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        days = (now - start_date).days
        if days < 0:
            return None
        return round(days / 365.25, 1)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _enrich_quote(q: dict) -> dict:
    """Extract applicable quote fields with clean column names."""
    sym = q.get("symbol")
    data = {
        "Symbol": sym,
        "Exchange": q.get("exchange") + " - " + q.get("exchDisp"),
        "Name": q.get("longname") or q.get("shortname"),
        "Currency": None,
        "Type": q.get("typeDisp") or q.get("quoteType"),
        "Years Exist": None,
    }
    if sym:
        info_result = yf_ticket_info(sym)

        if info_result.error is None and isinstance(info_result.data, dict):
            info = info_result.data
            clean_name = (
                q.get("longname")
                or info.get("longName")
                or info.get("displayName")
                or q.get("shortname")
                or info.get("shortName")
            )
            if clean_name:
                data["Name"] = clean_name

            data["Currency"] = info.get("currency")

            ft_ms = info.get("firstTradeDateMilliseconds")
            f_inc = info.get("fundInceptionDate")
            ts = (ft_ms / 1000.0) if ft_ms is not None else f_inc

            data["Years Exist"] = _calculate_years_in_existence(ts)

    return data


def get_quotes_by_symbol(search_input: str):
    """Search for a symbol and return quotes with applicable fields, filtering out results without years."""
    if not search_input or not search_input.strip():
        return None

    search_result = search_yf(search_input)

    if search_result.error is None and isinstance(search_result.data, dict):
        quotes = search_result.data.get("quotes", None)
        if quotes and len(quotes) > 0:
            enriched_quotes = [_enrich_quote(q) for q in quotes]

            df = pd.DataFrame(enriched_quotes)

            target_cols = [
                "Symbol",
                "Exchange",
                "Name",
                "Currency",
                "Type",
                "Years Exist",
            ]
            applicable_cols = [c for c in target_cols if c in df.columns]
            df = df[applicable_cols]

            # Drop results without years in existence
            if "Years Exist" in df.columns:
                df = df.dropna(subset=["Years Exist"])

            if not df.empty and "Symbol" in df.columns:
                df = df.set_index("Symbol")
                return df

    return None


def get_history_options(history_length: int, number_of_years: float, points_per_year: float):
    """Get the history options"""

    year_check = [
        i
        for i in ([10 * j for j in range(int(number_of_years / 10), 0, -1)] + [5, 3, 1])
        if 0 < round(i * points_per_year) < history_length
    ]

    max_years = round(number_of_years, 1)
    if max_years.is_integer():
        max_years = int(max_years)
    max_label = f"{max_years} {'Year' if max_years == 1 else 'Years'} (max)"
    return {max_label: history_length, **{
        f"{years} {'Year' if years == 1 else 'Years'}": round(years * points_per_year)
        for years in year_check
    }}


def generate_asset_base_value(asset_history: pd.DataFrame, fx_history: pd.Series):
    """Generate the base value for an asset"""

    asset_history["fx_history"] = fx_history

    asset_history["base_value"] = asset_history["value"] * \
        asset_history["fx_history"]

    asset_history.dropna(subset=["base_value"], inplace=True)

    return asset_history


def get_exp_fitted_data(y: list[float] | np.ndarray | pd.Series | Any):
    """Fit the value data to an exponential curve."""
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
    min_points = max(min_points, 2)

    result = pd.Series(index=series.index, dtype=float)

    for i in range(len(series)):
        if i < min_points:
            result.iloc[i] = series.iloc[i]
            continue

        # Get all data up to current point
        historical_data = series.iloc[: i + 1]
        x = np.arange(len(historical_data))
        y = historical_data.values

        # Fit exponential
        y_log = np.log(y)
        z_exp = np.polyfit(x, y_log, 1)
        p_exp = np.poly1d(z_exp)

        # Get the fitted value for the current point
        result.iloc[i] = np.exp(p_exp(len(historical_data) - 1))

    return result


def append_fitted_data(history_data: pd.DataFrame, selected_period: int, col_to_fit="base_value"):
    """Append the fitted data to the history data."""

    period_history_data = history_data.copy().tail(selected_period)

    period_history_data["fitted"] = get_exp_fitted_data(
        period_history_data[col_to_fit].values)

    days = (period_history_data.index[-1] - period_history_data.index[0]).days

    cagr = (period_history_data[col_to_fit].iloc[-1] / period_history_data[col_to_fit].iloc[0]) ** (
        1 / (days / 365.25)
    ) - 1
    cagr_fitted = (period_history_data["fitted"].iloc[-1] / period_history_data["fitted"].iloc[0]) ** (
        1 / (days / 365.25)
    ) - 1

    base_over_under = (
        period_history_data[col_to_fit].iloc[-1] -
        period_history_data["fitted"].iloc[-1]
    ) / period_history_data["fitted"].iloc[-1]

    period_history_data["Date"] = period_history_data.index

    return period_history_data, cagr, cagr_fitted, base_over_under


def get_period_returns(history: pd.Series, points_per_year: float) -> pd.DataFrame:
    """Calculate rolling returns using the history's average observation frequency."""
    returns = {}
    windows = [
        ("1D", 1),
        ("1W", max(1, round(points_per_year * 7 / 365.25))),
        ("1M", max(1, round(points_per_year / 12))),
        ("1Q", max(1, round(points_per_year / 4))),
        ("1Y", max(1, round(points_per_year))),
    ]
    for label, points in windows:
        previous = history.shift(points)
        returns[label] = history.div(previous.where(previous > 0)).sub(1)
    return pd.DataFrame(returns, index=history.index)


def create_asset_info_df(asset_info: dict) -> list[list[str]]:
    """Return the applicable identity and profile fields for an asset."""
    common_fields = [
        {"key": "longName", "label": "Name"},
        {"key": "quoteType", "label": "Quote Type"},
        {"key": "regularMarketPrice", "label": "Current Price"},
        {"key": "currency", "label": "Currency"},
        {"key": "exchange", "label": "Exchange"},
        {"key": "website", "label": "Website"},
    ]
    equity_fields = [
        {"key": "sector", "label": "Sector"},
        {"key": "industry", "label": "Industry"},
        {"key": "country", "label": "Country"},
        {"key": "marketCap", "label": "Market Cap"},
        {"key": "revenueGrowth", "label": "Revenue Growth (YoY)"},
        {"key": "earningsGrowth", "label": "Earnings Growth (YoY)"},
        {"key": "profitMargins", "label": "Profit Margin"},
        {"key": "forwardPE", "label": "Forward P/E"},
        {"key": "dividendYield", "label": "Dividend Yield"},
        {"key": "52WeekChange", "label": "Price Growth (1 Year)"},
        {"key": "fiftyTwoWeekRange", "label": "52-Week Range"},
        {"key": "firstTradeDateMilliseconds", "label": "Years Since First Trade"},
        {"key": "longBusinessSummary", "label": "Business Summary"},
    ]
    etf_fields = [
        {"key": "fundFamily", "label": "Fund Family"},
        {"key": "fundInceptionDate", "label": "Fund Inception Date"},
        {"key": "totalAssets", "label": "Fund Assets"},
        {"key": "fiftyTwoWeekChangePercent", "label": "Price Growth (1 Year)"},
        {"key": "fiftyTwoWeekRange", "label": "52-Week Range"},
        {"key": "averageVolume", "label": "Average Daily Volume"},
    ]

    quote_type = str(asset_info.get("quoteType", "")).upper()
    keys_to_display = common_fields
    if quote_type == "EQUITY":
        keys_to_display += equity_fields
    elif quote_type == "ETF":
        keys_to_display += etf_fields

    def format_value(key: str, value: Any) -> str:
        if key == "fundInceptionDate":
            return datetime.datetime.fromtimestamp(
                value, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        if key == "regularMarketPrice" and isinstance(value, (int, float)):
            currency = asset_info.get("currency", "")
            return f"{value:,.2f} {currency}".rstrip()
        if key in {"marketCap", "totalAssets"} and isinstance(value, (int, float)):
            for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M")):
                if abs(value) >= divisor:
                    currency = (
                        asset_info.get("financialCurrency")
                        if key == "marketCap"
                        else asset_info.get("currency")
                    ) or ""
                    return f"{value / divisor:.2f}{suffix} {currency}".rstrip()
            return f"{value:,.0f}"
        if key in {"revenueGrowth", "earningsGrowth", "profitMargins", "52WeekChange"}:
            return f"{value:.1%}"
        if key == "fiftyTwoWeekChangePercent":
            return f"{value:.1f}%"
        if key == "dividendYield":
            return f"{value:.2f}%"
        if key == "forwardPE":
            return f"{value:.1f}"
        if key == "averageVolume":
            return f"{value:,.0f}"
        if key == "firstTradeDateMilliseconds":
            years = _calculate_years_in_existence(value / 1000)
            return f"{years:.1f} years" if years is not None else ""
        return str(value)

    asset_info_rows = [
        [item["label"], format_value(item["key"], asset_info[item["key"]])]
        for item in keys_to_display
        if asset_info.get(item["key"]) not in (None, "")
    ]

    current_price = asset_info.get("regularMarketPrice")
    if current_price is None:
        current_price = asset_info.get("currentPrice")
    range_low = asset_info.get("fiftyTwoWeekLow")
    range_high = asset_info.get("fiftyTwoWeekHigh")
    two_hundred_day_average = asset_info.get("twoHundredDayAverage")

    if all(isinstance(value, (int, float)) for value in (current_price, range_low, range_high)) and range_high > range_low:
        range_width = (range_high - range_low) / ((range_high + range_low) / 2)
        range_position = (current_price - range_low) / (range_high - range_low)
        asset_info_rows.extend([
            ["52-Week Range Position", f"{range_position:.1%}"],
            ["52-Week Range Width", f"{range_width:.1%}"],
        ])

    if isinstance(current_price, (int, float)) and isinstance(two_hundred_day_average, (int, float)) and two_hundred_day_average:
        price_vs_two_hundred_day_average = current_price / two_hundred_day_average - 1
        asset_info_rows.append(
            ["Price vs 200-Day Average", f"{price_vs_two_hundred_day_average:.1%}"])

    return asset_info_rows


def get_trend_stats(periodic_asset_history_with_fit: pd.DataFrame, base_column="base_value"):
    """Get the trend stats"""
    days = periodic_asset_history_with_fit.shape[0]
    latest_base_value = periodic_asset_history_with_fit[base_column].iloc[-1]
    latest_fitted_value = periodic_asset_history_with_fit["fitted"].iloc[-1]
    oldest_base_value = periodic_asset_history_with_fit[base_column].iloc[0]
    oldest_fitted_value = periodic_asset_history_with_fit["fitted"].iloc[0]

    cagr = (latest_base_value / oldest_base_value) ** (1 / (days / 365.25)) - 1
    cagr_fitted = (latest_fitted_value /
                   oldest_fitted_value) ** (1 / (days / 365.25)) - 1

    base_over_under = latest_base_value / latest_fitted_value - 1

    return cagr, cagr_fitted, base_over_under


def get_trend_info(periodic_asset_history_with_fit: pd.DataFrame, base_column="base_value") -> pd.DataFrame:
    """Get the trend info DataFrame"""

    days = periodic_asset_history_with_fit.shape[0]
    latest_base_value = periodic_asset_history_with_fit[base_column].iloc[-1]
    latest_fitted_value = periodic_asset_history_with_fit["fitted"].iloc[-1]
    oldest_base_value = periodic_asset_history_with_fit[base_column].iloc[0]
    oldest_fitted_value = periodic_asset_history_with_fit["fitted"].iloc[0]

    cagr = (latest_base_value / oldest_base_value) ** (1 / (days / 365.25)) - 1
    cagr_fitted = (latest_fitted_value /
                   oldest_fitted_value) ** (1 / (days / 365.25)) - 1

    base_over_under = latest_base_value / latest_fitted_value - 1

    if "base_value" in periodic_asset_history_with_fit.columns:
        latest_value = periodic_asset_history_with_fit["base_value"].iloc[-1]
        oldest_value = periodic_asset_history_with_fit["base_value"].iloc[0]
        cagr_value = (latest_value / oldest_value) ** (1 / (days / 365.25)) - 1
    else:
        cagr_value = None

    return pd.DataFrame(
        data=[
            ["Sample Years", f"{days / 365.25:.1f}"],
            ["CAGR Base-fx", f"{(cagr-cagr_value):.1%}"],
            ["CAGR Base-inv",
                f"{(cagr_value):.1%}"] if cagr_value is not None else [],
            ["CAGR Base", f"{cagr:.1%}"],
            ["CAGR Fitted", f"{cagr_fitted:.1%}"],
            ["", ""],
            ["Date",
                periodic_asset_history_with_fit["Date"].iloc[-1].strftime("%Y-%m-%d")],
            ["Base Value",
                f"{periodic_asset_history_with_fit[base_column].iloc[-1]:,.2f}"],
            ["Fitted Value",
                f"{periodic_asset_history_with_fit['fitted'].iloc[-1]:,.2f}"],
            ["Base Over/Under", f"{base_over_under:.1%}"],
        ],
        columns=pd.Index(["Label", "Value"]),
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
    """Get the annual trend info DataFrame"""

    clean_df = periodic_asset_history_with_fit.dropna()

    if clean_df is None or clean_df.shape[0] == 0:
        return None, None

    mean = clean_df["annual_base_return"].mean()

    data_rows = [
        ["Sample Years", f"{clean_df.shape[0] / 365.25:.1f}"],
        ["Mean Annual Return", f"{mean:.1%}"],
        ["", ""],
        ["Date", clean_df["Date"].iloc[-1].strftime("%Y-%m-%d")],
        ["Current Annual Return",
            f"{clean_df['annual_base_return'].iloc[-1]:,.1%}"],
    ]

    return pd.DataFrame(data_rows, columns=pd.Index(["Label", "Value"])), mean


def get_fx_history_2(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""
    currency_symbol = target_currency + base_currency + "=X"
    crypto_symbol = target_currency + "-" + base_currency

    currency_rate_history = yf_ticket_history(currency_symbol).data
    crypto_rate_history = yf_ticket_history(crypto_symbol).data

    if isinstance(currency_rate_history, pd.Series) and not currency_rate_history.empty:
        return currency_rate_history, currency_symbol
    if isinstance(crypto_rate_history, pd.Series) and not crypto_rate_history.empty:
        return crypto_rate_history, crypto_symbol
    return None, None


def fetch_fx_rate_history(asset_currency: str, base_currency: str):
    """Fetch the fx rate for a given currency pair"""

    if asset_currency == base_currency:
        return pd.Series(1, index=pd.date_range(start="1950-01-01", end=pd.Timestamp.today(), freq="D"))

    currency_ticker_name = asset_currency + base_currency + "=X"

    fx_result = yf_ticket_history(currency_ticker_name)

    if fx_result.error is not None:
        return fx_result.error

    fx_history = fx_result.data.copy()
    fx_history.index = pd.DatetimeIndex(fx_history.index).tz_localize(None)

    return fx_history


def ticker_yf_history(symbol: str):
    """Fetch the asset info for a given symbol"""
    ticker_result = yf_ticket_history(symbol)

    if ticker_result.error is not None:
        return None

    ticker_history = ticker_result.data

    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)

    return ticker_history.to_frame(name="value")


def localize_and_fill(ticker_history: pd.DataFrame | pd.Series):
    """Localize and fill the ticker history"""
    if ticker_history.shape[0] == 0:
        return None

    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)
    ticker_history = ticker_history.resample("D").ffill()

    return ticker_history


def calculate_financial_metrics(df: pd.DataFrame, value_col: str, fitted_col: str, return_col: str):
    """Show the stats for the given DataFrame"""
    start_date = df.index[0]
    last_date = df.index[-1]
    actual_days_duration = (last_date - df.index[0]).days
    actual_years_duration = actual_days_duration / 365.25
    avg_points_per_year = df.shape[0] / actual_years_duration

    cagr = (df[value_col].iloc[-1] / df[value_col].iloc[0]
            ) ** (1 / actual_years_duration) - 1
    cagr_fitted = (df[fitted_col].iloc[-1] /
                   df[fitted_col].iloc[0]) ** (1 / actual_years_duration) - 1

    over_under = (df[value_col].iloc[-1] -
                  df[fitted_col].iloc[-1]) / df[fitted_col].iloc[-1]

    daily_returns_mean = df[return_col].mean()
    daily_return_std = df[return_col].std()

    annualized_returns_to_risk_ratio = (
        daily_returns_mean / daily_return_std) * np.sqrt(avg_points_per_year)

    annualized_return = (1 + daily_returns_mean) ** avg_points_per_year - 1
    annualized_risk = daily_return_std * np.sqrt(avg_points_per_year)

    return {
        metrics["start_date"].key: start_date,
        metrics["last_date"].key: last_date,
        metrics["actual_years_duration"].key: actual_years_duration,
        metrics["cagr"].key: cagr,
        metrics["cagr_fitted"].key: cagr_fitted,
        metrics["over_under"].key: over_under,
        metrics["annualized_return"].key: annualized_return,
        metrics["annualized_risk"].key: annualized_risk,
        metrics["annualized_returns_to_risk_ratio"].key: annualized_returns_to_risk_ratio,
    }


def fully_analyze_symbol(symbol: str, base_currency: str, years: int) -> Result[tuple]:
    """Fetch the asset info for a given symbol"""

    ###############################
    # Fetch from Yahoo Finance API

    # Get asset info to get the currency
    ticker_info_result = yf_ticket_info(symbol)
    if ticker_info_result.error is not None:
        return Result(data=None, error=ticker_info_result.error)

    asset_currency = ticker_info_result.data.get("currency")
    if not asset_currency:
        return Result(data=None, error=f"Asset currency for {symbol} is not defined in the ticker info")

    # Get asset history
    ticker_history_result = yf_ticket_history(symbol)
    if ticker_history_result.error is not None:
        return Result(data=None, error=ticker_history_result.error)
    ticker_history = ticker_history_result.data
    if ticker_history.shape[0] == 0:
        return Result(data=None, error=f"Retrieved ticker history is empty for {symbol}")

    # Get the fx rate history
    fx_history_result = get_fx_history(base_currency, asset_currency)
    if fx_history_result.error is not None:
        return Result(data=None, error=fx_history_result.error)
    fx_history = fx_history_result.data
    if fx_history.shape[0] == 0:
        return Result(
            data=None, error=f"Retrieved fx history is empty for {base_currency} and {asset_currency}"
        )

    ###############################
    # Localize and Filter data window
    ticker_history.index = pd.DatetimeIndex(
        ticker_history.index).tz_localize(None)
    fx_history.index = pd.DatetimeIndex(fx_history.index).tz_localize(None)

    start_date = ticker_history.index[-1] - pd.DateOffset(years=years)
    ticker_history = ticker_history[ticker_history.index >= start_date]

    ###############################
    # Trade Value
    trade_value_col = "Trade Value"
    trade_value_return_col = "Trade Value Return"
    trade_value_fitted_col = "Trade Value Fitted"

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
    fx_col = "FX Rate"
    home_value_col = "Home Value"
    home_value_return_col = "Home Value Return"
    home_value_fitted_col = "Home Value Fitted"

    # Rename the fx_history series properly
    fx_history_renamed = fx_history.copy()
    fx_history_renamed.name = fx_col

    home_df = pd.concat(
        [trade_df[trade_value_col], fx_history_renamed], axis=1, join="inner")

    home_df[home_value_col] = home_df[trade_value_col] * home_df[fx_col]
    home_df[home_value_return_col] = home_df[home_value_col].pct_change()
    home_df[home_value_fitted_col] = get_exp_fitted_data(
        home_df[home_value_col].values)

    home_df.drop(columns=[trade_value_col], inplace=True)

    trade_metrics = calculate_financial_metrics(
        trade_df, trade_value_col, trade_value_fitted_col, trade_value_return_col
    )

    home_metrics = calculate_financial_metrics(
        home_df, home_value_col, home_value_fitted_col, home_value_return_col)

    return Result(data=(trade_df, trade_metrics, home_df, home_metrics), error=None)


def retrieve_sector_data(home_currency: str, data_years: int):
    """Retrieve the sector data"""
    sectors_data = list(
        sorted(
            [r.data if r.error is None else r.error for r in (
                sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS)],
            key=lambda x: (
                x["overview"]["market_weight"]
                if isinstance(x, dict) and "overview" in x and "market_weight" in x["overview"]
                else -99
            ),
            reverse=True,
        )
    )

    # Process sectors and add etf_metrics
    processed_sectors = []
    for sector in sectors_data:
        if not isinstance(sector, dict):
            processed_sectors.append(sector)
            continue

        top_etfs = sector.get("top_etfs")
        if top_etfs is not None and isinstance(top_etfs, dict) and len(list(top_etfs.keys())) > 0:

            first_etf = list(top_etfs.keys())[0]
            info = yf_ticket_info(first_etf).data
            etf_result = fully_analyze_symbol(
                first_etf, home_currency, data_years)

            if etf_result.error is not None:
                processed_sectors.append(sector)
                continue

            _, etf_trade_metrics, _, etf_home_metrics = etf_result.data

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
            "Sector": sector["name"],
            "Market Weight": sector["overview"]["market_weight"],
            "ETF": sector["etf_metrics"]["top_etf"],
            # "ETF Summary":   sector["etf_metrics"]["etf_info"]["longBusinessSummary"],
            "Trade Over/Under": sector["etf_metrics"]["etf_trade_metrics"]["over_under"],
            "Trade CAGR": sector["etf_metrics"]["etf_trade_metrics"]["cagr"],
            "Trade CAGR Fitted": sector["etf_metrics"]["etf_trade_metrics"]["cagr_fitted"],
            "Trade Annualized Return": sector["etf_metrics"]["etf_trade_metrics"]["annualized_return"],
            "Trade Annualized Risk": sector["etf_metrics"]["etf_trade_metrics"]["annualized_risk"],
            "Trade Return/Risk Ratio": sector["etf_metrics"]["etf_trade_metrics"]["annualized_returns_to_risk_ratio"],
            "Reference Date": sector["etf_metrics"]["etf_home_metrics"]["last_date"],
            "Sample Years": sector["etf_metrics"]["etf_home_metrics"]["actual_years_duration"],
        }
        for sector in processed_sectors
        if isinstance(sector, dict) and "etf_metrics" in sector
    ]

    total_row = {
        "Sector": "Total",
        "Market Weight": sum(sector["Market Weight"] for sector in data),
        "Trade CAGR": sum(sector["Trade CAGR"] * sector["Market Weight"] for sector in data),
    }

    return sectors_data, pd.DataFrame(data + [total_row])


def fetch_multiple_sectors_data():
    """Fetch the sector data for the given keys"""
    sector_results = [sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS]

    sector_errors = [r.error for r in sector_results if r.error is not None]
    sector_data = [r.data for r in sector_results if r.error is None]

    sector_data = sorted(
        sector_data,
        key=lambda x: (
            x["overview"]["market_weight"]
            if isinstance(x, dict) and "overview" in x and "market_weight" in x["overview"]
            else -99
        ),
        reverse=True,
    )

    # check sum of sector weights
    sum_of_weights = sum(sector["overview"]["market_weight"]
                         for sector in sector_data)
    if round(sum_of_weights, 2) != 1:
        logger.warning("Sum of sector weights is not 1: %s", sum_of_weights)

    return sector_data, sector_errors


def get_period_info(p_history: pd.DataFrame):
    """Get the period info for the given history"""
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
    """Get the first and last values from the history and fitted history"""

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
    """Get the tickers data extended"""

    fitted_history = history.apply(get_exp_fitted_data)
    trend_deviation = history / fitted_history - 1

    if not isinstance(fitted_history, pd.DataFrame):
        raise ValueError("Fitted history is not a DataFrame")

    if not isinstance(trend_deviation, pd.DataFrame):
        raise ValueError("CAGR error is not a DataFrame")

    first_value, last_value, first_fitted_value, last_fitted_value = get_first_last_values(
        history, fitted_history)

    _, _, number_of_years, _, _ = get_period_info(history)

    cagr = (last_value / first_value) ** (1 / number_of_years) - 1
    cagr_fitted = (last_fitted_value /
                   first_fitted_value) ** (1 / number_of_years) - 1
    over_under = (last_value - last_fitted_value) / last_fitted_value

    if not isinstance(cagr, pd.Series):
        raise ValueError("Error: CAGR is not a Series")

    if not isinstance(cagr_fitted, pd.Series):
        raise ValueError("Error: CAGR fitted is not a Series")

    if not isinstance(over_under, pd.Series):
        raise ValueError("Error: Over/Under is not a Series")

    trend_deviation_rmse = np.sqrt((trend_deviation**2).mean())

    if not isinstance(trend_deviation_rmse, pd.Series):
        raise ValueError("Error: CAGR error RMSE is not a Series")

    trend_deviation_z_score = (
        trend_deviation - trend_deviation.mean()) / trend_deviation.std()

    return fitted_history, trend_deviation, cagr, cagr_fitted, over_under, trend_deviation_rmse, trend_deviation_z_score


def get_log_returns(p_history: pd.DataFrame):
    """Get the log returns for the given history"""

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

    return (
        cumulative_log_returns,
        daily_log_returns,
        monthly_log_returns,
        yearly_log_returns,
        mean_log_daily_returns,
        std_log_daily_returns,
        mean_log_monthly_returns,
        std_log_monthly_returns,
        mean_log_yearly_returns,
        std_log_yearly_returns,
    )


class MultiAsset:
    """Multi-asset analysis class"""

    def __init__(
        self, symbols: list[str], period: str = "max", debug: bool = False, weights: list[float] | None = None
    ):
        """Initialize MultiAsset with symbols and calculate metrics"""
        self.symbols = symbols
        self.period = period
        self.debug = debug

        # Fetch data from Yahoo Finance
        tickers_result = tickers_yf(symbols, period)
        if tickers_result.error is not None:
            raise ValueError(tickers_result.error)

        tickers_data = tickers_result.data
        if not isinstance(tickers_data, dict):
            raise ValueError("Tickers data is not a dictionary")

        history = tickers_data["history"]
        if not isinstance(history, pd.DataFrame):
            raise ValueError("History is not a DataFrame")

        # Fetch info for each symbol
        info_results = [yf_ticket_info(symbol) for symbol in history.columns]
        error_list = [r.error for r in info_results if r.error is not None]
        if error_list:
            raise ValueError(
                f"Error: Failed to retrieve ticker info for some symbols: {error_list}")
        clean_info_list = [r.data for r in info_results if r.error is None]

        currencies = pd.Series(
            data=[info.get("currency", "") for info in clean_info_list],
            index=[info.get("symbol", "") for info in clean_info_list],
        )

        long_names = pd.Series(
            data=[info.get("longName", "") for info in clean_info_list],
            index=[info.get("symbol", "") for info in clean_info_list],
        )

        types = pd.Series(
            data=[info.get("typeDisp", "") for info in clean_info_list],
            index=[info.get("symbol", "") for info in clean_info_list],
        )

        long_business_summaries = pd.Series(
            data=[info.get("longBusinessSummary", "")
                  for info in clean_info_list],
            index=[info.get("symbol", "") for info in clean_info_list],
        )

        # Set default weights if not provided
        if weights is None:
            weights = [1 / len(history.columns)] * len(history.columns)

        if len(weights) != len(history.columns):
            raise ValueError(
                "Error: Sector weights length does not match symbols length")

        # Make sure the weights sum to 1
        weights_sum = sum(weights)
        self.weights = [
            w / weights_sum for w in weights] if weights_sum > 0 else weights

        # Create weighted total
        history["TOTAL"] = history.mul(self.weights, axis=1).sum(axis=1)

        # Calculate fitted data and metrics
        (
            fitted_history,
            trend_deviation,
            cagr,
            cagr_fitted,
            over_under,
            trend_deviation_rmse,
            trend_deviation_z_score,
        ) = get_history_exp_fit(history)

        # Calculate log returns
        (
            cumulative_log_returns,
            daily_log_returns,
            monthly_log_returns,
            yearly_log_returns,
            mean_log_daily_returns,
            std_log_daily_returns,
            mean_log_monthly_returns,
            std_log_monthly_returns,
            mean_log_yearly_returns,
            std_log_yearly_returns,
        ) = get_log_returns(history)

        # Debug output
        if self.debug:
            self._print_debug_info(
                history,
                fitted_history,
                trend_deviation,
                daily_log_returns,
                cumulative_log_returns,
                monthly_log_returns,
                yearly_log_returns,
                cagr,
                cagr_fitted,
                over_under,
                trend_deviation_rmse,
                mean_log_daily_returns,
                std_log_daily_returns,
                mean_log_monthly_returns,
                std_log_monthly_returns,
                mean_log_yearly_returns,
                std_log_yearly_returns,
            )

        # Create symbol metrics DataFrame
        weights_series = pd.Series(
            self.weights + [1], index=history.columns)  # add 1 for TOTAL

        self.symbol_metrics = pd.concat(
            [
                currencies,
                long_names,
                types,
                long_business_summaries,
                cagr,
                cagr_fitted,
                over_under,
                trend_deviation_rmse,
                weights_series,
                mean_log_daily_returns,
                std_log_daily_returns,
                mean_log_monthly_returns,
                std_log_monthly_returns,
                mean_log_yearly_returns,
                std_log_yearly_returns,
            ],
            axis=1,
            keys=[
                "currencies",
                "long_names",
                "types",
                "long_business_summaries",
                "cagr",
                "cagr_fitted",
                "over_under",
                "trend_deviation_rmse",
                "weights",
                "mean_log_daily_returns",
                "std_log_daily_returns",
                "mean_log_monthly_returns",
                "std_log_monthly_returns",
                "mean_log_yearly_returns",
                "std_log_yearly_returns",
            ],
        )

        # Create timeseries data DataFrame
        self.timeseries_data = pd.concat(
            objs=[
                history,
                fitted_history,
                trend_deviation,
                trend_deviation_z_score,
                daily_log_returns,
                cumulative_log_returns,
                monthly_log_returns,
                yearly_log_returns,
            ],
            axis=1,
            keys=[
                "history",
                "fitted_history",
                "trend_deviation",
                "trend_deviation_z_score",
                "daily_log_returns",
                "cumulative_log_returns",
                "monthly_log_returns",
                "yearly_log_returns",
            ],
        )

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return symbol metrics and timeseries data"""
        return self.symbol_metrics, self.timeseries_data

    def get_period_info(self) -> dict[str, Any]:
        """Extract and return period information from timeseries data with type checking"""

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
            "first_date": first_date,
            "last_date": last_date,
            "total_days": total_days,
            "sample_years": sample_years,
            "data_points": data_points,
            "frequency": frequency,
        }

    def create_asset_analysis_chart(self, selected_asset: str) -> go.Figure:
        """Create dual-axis chart for selected asset with history, fitted history, and z-score"""
        # Validate asset exists
        available_assets = self.timeseries_data.columns.get_level_values(
            1).unique().tolist()
        if selected_asset not in available_assets:
            raise ValueError(
                f"Asset '{selected_asset}' not found. Available assets: {available_assets}")

        # Prepare data for selected asset
        history_data = self.timeseries_data[(
            "history", selected_asset)].dropna()
        fitted_data = self.timeseries_data[(
            "fitted_history", selected_asset)].dropna()
        z_score_data = self.timeseries_data[(
            "trend_deviation_z_score", selected_asset)].dropna()

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add history line (primary axis)
        fig.add_trace(
            go.Scatter(
                x=history_data.index,
                y=history_data.values,
                mode="lines",
                name="History",
                line={"color": "#1f77b4", "width": 2},
            ),
            secondary_y=False,
        )

        # Add fitted history line (primary axis)
        fig.add_trace(
            go.Scatter(
                x=fitted_data.index,
                y=fitted_data.values,
                mode="lines",
                name="Fitted History",
                line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            ),
            secondary_y=False,
        )

        # Add z-score line (secondary axis)
        fig.add_trace(
            go.Scatter(
                x=z_score_data.index,
                y=z_score_data.values,
                mode="lines",
                name="Trend Deviation Z-Score",
                line={"color": "#d62728", "width": 1.5},
                opacity=0.7,
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title=f"Asset Analysis: {selected_asset}",
            xaxis_title="Date",
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom",
                    "y": 1.02, "xanchor": "right", "x": 1},
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="Value", secondary_y=False)
        fig.update_yaxes(title_text="Z-Score", secondary_y=True)

        # Add horizontal line at z-score = 0
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      opacity=0.5, secondary_y=True)

        return fig

    def _print_debug_info(
        self,
        history,
        fitted_history,
        trend_deviation,
        daily_log_returns,
        cumulative_log_returns,
        monthly_log_returns,
        yearly_log_returns,
        cagr,
        cagr_fitted,
        over_under,
        trend_deviation_rmse,
        mean_log_daily_returns,
        std_log_daily_returns,
        mean_log_monthly_returns,
        std_log_monthly_returns,
        mean_log_yearly_returns,
        std_log_yearly_returns,
    ):
        """Print debug information"""
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
