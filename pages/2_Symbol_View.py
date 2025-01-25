# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""
import re
from typing import List
import datetime
import yfinance as yf
import streamlit as st
import altair as alt
import pandas as pd
from utilities.utilities import AssetDetails, fetch_asset_info_and_history, create_asset_info_df, \
    fetch_fx_rate_history, generate_asset_base_value, append_fitted_data
from utilities.constants import BASE_CURRENCY_OPTIONS

print(f"\n--- Now: {datetime.datetime.now()} ---\n")


def convert_symbol_to_yf_format(symbol):
    """Convert the symbol to Yahoo Finance format."""
    # remove dots at the start and end of the string and consecutive dots
    # replace all characters except letters and dots with empty string
    if symbol:
        return re.sub(r'\.{2,}', '.', re.sub(r'[^A-Za-z\.]', '', symbol)).strip(".").upper()
    else:
        return ""


@st.cache_data
def validate_symbol_exists(symbol):
    """Validate if the symbol exists in Yahoo Finance."""
    if yf.Ticker(symbol).history(period="1d").empty:
        st.error("Symbol not found.")
        return False
    else:
        return True


st.title("Symbol Information")


symbol_name = convert_symbol_to_yf_format(
    st.text_input(label="Enter a symbol", value="VUSA.L", key="symbol_input"))

if len(symbol_name) > 0:

    st.subheader(f"Symbol: {symbol_name}")

    if validate_symbol_exists(symbol_name):

        asset_info, full_asset_history = fetch_asset_info_and_history(
            symbol_name)

        asset_info_table = create_asset_info_df(asset_info)

        st.dataframe(asset_info_table, hide_index=True,
                     use_container_width=True)

        st.markdown("#### Trend Analysis")

        col1, col2 = st.columns(2)

        with col1:
            base_currency = st.selectbox(
                "Select currency", BASE_CURRENCY_OPTIONS, key="currency_input")

        # In this case we have one asset with a position of 1.0
        assets_positions: List[AssetDetails] = [
            {"symbol": symbol_name, "position": 1.0}]

        # Fetch the fx rate history for the asset currency
        full_fx_rate_history = fetch_fx_rate_history(
            asset_info['currency'], base_currency)

        # Add the fx rate history to the asset history
        full_asset_base_history = generate_asset_base_value(
            full_asset_history, full_fx_rate_history)

        # Fetch the historical data for the asset and info

        period_options = [f"Max ({round(full_asset_base_history['base_value'].shape[0]/365.25, 1)} years)"] + \
            [str(period) + (" Year" if period == 1 else " Years")
             for period in [10, 5, 3, 1] if full_asset_base_history['base_value'].shape[0] > period * 365.25]

        with col2:
            selected_period = st.selectbox("Select period", period_options)

        # Generate the data for the selected period
        periodic_asset_history_with_fit, cagr, cagr_fitted, base_over_under = append_fitted_data(
            full_asset_base_history, selected_period)

        # Display the CAGR and CAGR for the fitted data
        trend_info_df = pd.DataFrame(
            columns=['Label', 'Value'],
            data=[
                ['Sample Years', f"{
                    periodic_asset_history_with_fit.shape[0] / 365.25:.1f}"],
                ['CAGR Base', f"{cagr:.1%}"],
                ['CAGR Fitted', f"{cagr_fitted:.1%}"],
                ['', ''],
                ['Date',
                    periodic_asset_history_with_fit['Date'].iloc[-1].strftime('%Y-%m-%d')],
                ['Base Value', f"{
                    periodic_asset_history_with_fit['base_value'].iloc[-1]:,.2f}"],
                ['Fitted Value', f"{
                    periodic_asset_history_with_fit['fitted'].iloc[-1]:,.2f}"],
                ['Base Over/Under', f"{base_over_under:.1%}"]]
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(trend_info_df, hide_index=True)
            st.write("*CAGR: Compound Annual Growth Rate*")

        # Create the line chart

        y_axis_padding = 0.1 * \
            (periodic_asset_history_with_fit['base_value'].max() -
             periodic_asset_history_with_fit['base_value'].min())

        y_axis_start = periodic_asset_history_with_fit[['base_value',
                                                        'fitted']].min().min() - y_axis_padding

        y_axis_end = periodic_asset_history_with_fit[['base_value',
                                                      'fitted']].max().max() + y_axis_padding

        chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
            x=alt.X('Date:T', title=None),
            y=alt.Y('base_value:Q', scale=alt.Scale(
                domain=[y_axis_start, y_axis_end]), title=f'Base Value {cagr:.1%}'),
            color=alt.value('#14B3EB')  # Set custom color
        )

        fitted_chart = alt.Chart(periodic_asset_history_with_fit).mark_line().encode(
            x=alt.X('Date:T', title=None),
            y=alt.Y('fitted:Q', scale=alt.Scale(
                domain=[y_axis_start, y_axis_end]), title=f'Fitted Value {cagr_fitted:.1%}'),
            color=alt.value('#EB4C14')  # Set custom color
        )

        with col2:
            st.altair_chart(
                (chart + fitted_chart).properties(height=425), use_container_width=True)

        # lets show a chart of daily annual returns
        st.markdown("#### Daily Annual Returns")

        daily_returns_chart = alt.Chart(periodic_asset_history_with_fit.dropna()).mark_line().encode(
            x=alt.X('Date:T', title=None),
            y=alt.Y('annual_base_return:Q', title='Annual Return',
                    axis=alt.Axis(format='.1%')),
            color=alt.condition(
                alt.datum.annual_base_return > 0,
                alt.value("#14B3EB"),
                alt.value("#EB4C14")
            )
        )

        geometric_mean = (1 + periodic_asset_history_with_fit.dropna()['annual_base_return']).prod() \
            ** (1/periodic_asset_history_with_fit.dropna().shape[0]) \
            - 1

        # Make the line of mean
        mean_line = alt.Chart(pd.DataFrame({'mean': [geometric_mean]})).mark_rule(
            color='purple', size=2).encode(y='mean:Q')

        # Zero line
        zero_line = alt.Chart(pd.DataFrame({'zero': [0]})).mark_rule(
            color='red', size=1).encode(y='zero:Q')

        annual_returns_info = pd.DataFrame(
            columns=['Label', 'Value'],
            data=[
                ['Sample Years', f"{
                    periodic_asset_history_with_fit.dropna().shape[0] / 365.25:.1f}"],
                ['Mean Annual Return', f"{geometric_mean:.1%}"],
                ['', ''],
                ['Date',
                    periodic_asset_history_with_fit['Date'].iloc[-1].strftime('%Y-%m-%d')],
                ['Current Annual Return', f"{
                    periodic_asset_history_with_fit['annual_base_return'].iloc[-1]:,.1%}"],
            ]
        )

        annual_col1, annual_col2 = st.columns([1, 2])

        with annual_col1:
            st.dataframe(annual_returns_info, hide_index=True)

        with annual_col2:
            st.altair_chart((daily_returns_chart+mean_line +
                            zero_line).properties(height=250),
                            use_container_width=True)
