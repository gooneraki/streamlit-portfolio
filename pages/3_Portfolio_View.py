# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""
import datetime
import json
import pandas as pd
import streamlit as st
from utilities.utilities import generate_asset_base_value, append_fitted_data, get_trend_info, get_history_options, \
    fetch_fx_rate_history, ticker_yf_history
from utilities.constants import ASSETS_POSITIONS_DEFAULT, BASE_CURRENCY_OPTIONS
from utilities.go_charts import display_trend_go_chart
from utilities.app_yfinance import yf_ticket_info
from classes.asset_positions import AssetPosition, Portfolio


print(f"\n--- Portfolio view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="Portfolio View", layout="wide")

col1, col2 = st.columns([1, 2])
with col1:
    username = st.text_input(label="Enter username",
                             key="username_input", type="password")


def get_assets_positions():
    """ Get the assets positions """
    try:
        if username == st.secrets["DB_USERNAME"]:
            positions_dict = json.loads(st.secrets["ASSETS_POSITIONS_STR"])
            result = [AssetPosition(**position) for position in positions_dict]
            return result
        else:
            return ASSETS_POSITIONS_DEFAULT
    except Exception as err:
        print(f"Error: {err}")
        return ASSETS_POSITIONS_DEFAULT


assets_positions = get_assets_positions()


col1, col2 = st.columns([1, 2])
with col1:
    base_currency = st.selectbox(
        "Select base currency", BASE_CURRENCY_OPTIONS, key="base_currency_input")

portfolio = Portfolio(assets_positions, base_currency)

st.title("Portfolio Information")

first_date, last_date, number_of_points, number_of_days, number_of_years, points_per_year, points_per_month = portfolio.get_period_info()

# Display sampling period information
st.markdown("#### 📊 Data Sampling Period")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Start Date", first_date.strftime('%Y-%m-%d'))
with col2:
    st.metric("End Date", last_date.strftime('%Y-%m-%d'))
with col3:
    st.metric("Sample Period", f"{number_of_years:.1f} years")
with col4:
    st.metric("Data Points", f"{number_of_points:,}")

st.info(
    f"📈 **Analysis Coverage:** {number_of_days:,} days ({number_of_years:.1f} years) with {number_of_points:,} data points (avg {points_per_year:.0f} points/year)")

st.markdown("---")

st.markdown("#### Portfolio Composition")


st.dataframe(
    pd.DataFrame(portfolio.get_assets_metrics().drop(
        columns=['cagr', 'cagr_fitted', 'latest_weights'])),
    column_config={
        "position": st.column_config.NumberColumn(label="Position", format="localized"),
        "cagr_pct": st.column_config.NumberColumn(
            label="CAGR",
            format="%.1f%%",
            help="Compound Annual Growth Rate",
        ),
        "cagr_fitted_pct": st.column_config.NumberColumn(
            label="CAGR Fitted",
            format="%.1f%%",
            help="Compound Annual Growth Rate Fitted",
        ),
        "latest_value": st.column_config.NumberColumn(
            label="Latest Value",
            format="localized"
        ),
        "latest_fitted_value": st.column_config.NumberColumn(
            label="Latest Fitted Value",
            format="localized"
        ),
        "latest_weights_pct": st.column_config.NumberColumn(
            label="Latest Weights",
            format="%.1f%%"
        ),
    })


st.markdown("#### Portfolio Performance")

aggregate_df = pd.DataFrame()

for asset in assets_positions:

    asset_info = yf_ticket_info(asset.get_symbol())
    full_asset_history = ticker_yf_history(asset.get_symbol())

    # Fetch the fx rate history for the asset currency
    full_fx_rate_history = fetch_fx_rate_history(
        asset_info['currency'], base_currency)

    # Add the fx rate history to the asset history
    full_asset_base_history = generate_asset_base_value(
        full_asset_history, full_fx_rate_history)

    # Add the base value to the aggregate dataframe
    full_asset_base_history['base_value'] *= asset.get_position()

    aggregate_df = pd.concat([aggregate_df, full_asset_base_history[['base_value']].rename(
        columns={'base_value': asset.get_symbol()})], axis=1)

# Drop any rows with NaN values
aggregate_df.dropna(inplace=True)

# TODO : Backwards fill

# Calculate the portfolio value
aggregate_df['base_value'] = aggregate_df.sum(axis=1)


history_options = get_history_options(aggregate_df['base_value'].shape[0])

history_options_keys = list(history_options.keys())

col3, col4 = st.columns([1, 2])
with col3:
    selected_period_key = st.selectbox(
        "Select period",
        history_options_keys,
        index=history_options_keys.index("10 Years") if "10 Years" in history_options_keys else 0)
    selected_period_value = history_options[selected_period_key]

periodic_asset_history_with_fit, cagr, cagr_fitted, base_over_under = append_fitted_data(
    aggregate_df, selected_period_value)

col1, col2 = st.columns([1, 2])

with col1:
    trend_info_df = get_trend_info(periodic_asset_history_with_fit)
    st.dataframe(trend_info_df, hide_index=True)
    st.write("*CAGR: Compound Annual Growth Rate*")

# Create the line chart
with col2:
    value_fig = display_trend_go_chart(periodic_asset_history_with_fit)

    if value_fig is None:
        st.warning("No valid data to plot.")
    else:
        st.plotly_chart(value_fig, use_container_width=True)
