# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""
import datetime
import json
import pandas as pd
import streamlit as st
from utilities.utilities import generate_asset_base_value, append_fitted_data, get_trend_info, get_history_options
from utilities.constants import BASE_CURRENCY_OPTIONS
from utilities.go_charts import display_trend_go_chart
from utilities.app_yfinance import fetch_fx_rate_history, ticker_yf_history, yf_ticket_info


print(f"\n--- Portfolio view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="Portfolio View", layout="centered")

username = st.text_input(label="Enter username",
                         key="username_input", type="password")


assets_positions = json.loads(st.secrets["ASSETS_POSITIONS_STR"]) if username == st.secrets["DB_USERNAME"] else [
    {
        "symbol": "CSPX.L",
        "position": 1
    },
    {
        "symbol": "IUSE.L",
        "position": 1
    },
    {
        "symbol": "IWDE.L",
        "position": 1
    },
    {
        "symbol": "IWDA.L",
        "position": 1
    }
]


st.title("Portfolio Information")

st.markdown("#### Portfolio Composition")

assets_positions_df = pd.DataFrame(assets_positions)

st.dataframe(assets_positions_df, hide_index=True)

col1, col2 = st.columns([1, 2])
with col1:
    base_currency = st.selectbox(
        "Select base currency", BASE_CURRENCY_OPTIONS, key="base_currency_input")

st.markdown("#### Portfolio Performance")

aggregate_df = pd.DataFrame()

for asset in assets_positions:

    asset_info = yf_ticket_info(asset['symbol'])
    full_asset_history = ticker_yf_history(asset['symbol'])

    # Fetch the fx rate history for the asset currency
    full_fx_rate_history = fetch_fx_rate_history(
        asset_info['currency'], base_currency)

    # Add the fx rate history to the asset history
    full_asset_base_history = generate_asset_base_value(
        full_asset_history, full_fx_rate_history)

    # Add the base value to the aggregate dataframe
    full_asset_base_history['base_value'] *= asset['position']

    aggregate_df = pd.concat([aggregate_df, full_asset_base_history[['base_value']].rename(
        columns={'base_value': asset['symbol']})], axis=1)

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
