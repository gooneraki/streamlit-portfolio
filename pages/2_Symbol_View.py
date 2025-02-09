# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""

from typing import List
import datetime
import yfinance as yf
import streamlit as st
import altair as alt
import pandas as pd
from utilities.utilities import AssetDetails, fetch_asset_info_and_history, create_asset_info_df, \
    fetch_fx_rate_history, generate_asset_base_value, append_fitted_data, get_trend_info, \
    get_annual_returns_trend_info, display_trend_line_chart, fetch_asset_info
from utilities.constants import BASE_CURRENCY_OPTIONS

print(f"\n--- Now: {datetime.datetime.now()} ---\n")


@st.cache_data
def validate_symbol_exists(symbol):
    """Validate if the symbol exists in Yahoo Finance."""
    if yf.Ticker(symbol).history(period="1d").empty:
        st.error("Symbol not found.")
        return False
    else:
        return True


st.title("Symbol Information")

st.write("### Search symbols")

search_input = st.text_input("First result will be analyzed",
                             key="search_input",  placeholder="E.g. VUSA, CSPX, EQQQ, VWRL, AGGH, VFEM, VHYL")

search_results = yf.Search(search_input)

symbol_name = search_results.quotes[0]['symbol'] if search_results is not None and len(
    search_results.quotes) > 0 else None

st.write("##### Search results")

if search_results is not None and len(search_results.quotes) > 0:
    result_quotes = search_results.quotes

    result_quotes_df = pd.DataFrame(
        search_results.quotes).drop("index", axis=1)
    result_quotes_df.set_index('symbol', inplace=True)

    st.dataframe(result_quotes_df)


if symbol_name is not None:

    st.subheader(f"Symbol: {symbol_name}")

    asset_info_df = pd.DataFrame.from_dict(
        fetch_asset_info(symbol_name), orient='index', columns=['Value'])

    asset_info_df.index.name = "Attribute"

    st.info("Asset Information")

    with st.expander("Show all info"):
        st.dataframe(asset_info_df.dropna(), use_container_width=True)

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

        col1, col2 = st.columns([1, 2])

        with col1:
            trend_info_df = get_trend_info(periodic_asset_history_with_fit)
            st.dataframe(trend_info_df, hide_index=True)
            st.write("*CAGR: Compound Annual Growth Rate*")

        # Create the line chart
        with col2:
            display_trend_line_chart(periodic_asset_history_with_fit)

        # lets show a chart of daily annual returns
        st.markdown("#### Daily Annual Returns")

        annual_returns_info, mean = get_annual_returns_trend_info(
            periodic_asset_history_with_fit)

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

        # Make the line of mean
        mean_line = alt.Chart(pd.DataFrame({'mean': [mean]})).mark_rule(
            color='purple', size=2).encode(y='mean:Q')

        # Zero line
        zero_line = alt.Chart(pd.DataFrame({'zero': [0]})).mark_rule(
            color='red', size=1).encode(y='zero:Q')

        annual_col1, annual_col2 = st.columns([1, 2])

        with annual_col1:
            st.dataframe(annual_returns_info, hide_index=True)

        with annual_col2:
            st.altair_chart((daily_returns_chart+mean_line +
                            zero_line).properties(height=250),
                            use_container_width=True)
