import datetime
import pandas as pd
import streamlit as st
from utilities.utilities import fetch_asset_info_and_history, fetch_fx_rate_history, generate_asset_base_value, append_fitted_data
from utilities.constants import BASE_CURRENCY_OPTIONS
import json


print(f"\n--- Portfolio view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="Portfolio View", layout="centered")

username = st.text_input(label="Enter username",
                         key="username_input", type="password")
print(st.secrets["DB_USERNAME"])


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

    asset_info, full_asset_history = fetch_asset_info_and_history(
        asset["symbol"])

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
# Calculate the portfolio value
aggregate_df['Portfolio'] = aggregate_df.sum(axis=1)

period_options = [f"Max ({round(aggregate_df['Portfolio'].shape[0]/365.25, 1)} years)"] + \
    [str(period) + (" Year" if period == 1 else " Years")
        for period in [10, 5, 3, 1] if aggregate_df['Portfolio'].shape[0] > period * 365.25]

col3, col4 = st.columns([1, 2])
with col3:
    selected_period = st.selectbox("Select period", period_options)

periodic_asset_history_with_fit, cagr, cagr_fitted, base_over_under = append_fitted_data(
    aggregate_df, selected_period, 'Portfolio')

print(f"Portfolio CAGR: {cagr}")
print(f"Portfolio CAGR fitted: {cagr_fitted}")
print(f"Portfolio over/under: {base_over_under}")
st.line_chart(periodic_asset_history_with_fit[['Portfolio', 'fitted']])
