# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""

from typing import List
import datetime
import streamlit as st
import altair as alt
import pandas as pd
from utilities.utilities import AssetDetails, fetch_asset_history, create_asset_info_df, \
    fetch_fx_rate_history, generate_asset_base_value, append_fitted_data, get_trend_info, \
    get_annual_returns_trend_info, display_trend_line_chart, fetch_asset_info, search_symbol
from utilities.constants import BASE_CURRENCY_OPTIONS

print(f"\n--- Now: {datetime.datetime.now()} ---\n")


def reset_query_params(p_search_input: str):
    """Reset the query parameters."""
    if p_search_input is None:
        st.query_params.pop("symbol", None)
    else:
        st.query_params.update({"symbol": p_search_input})


st.title("Symbol Information")

st.write("### Search symbols")


with st.form(key="search_form"):
    search_input = st.text_input("First result will be analyzed",
                                 value=st.query_params.get("symbol"),
                                 max_chars=10,
                                 key="search_input",
                                 placeholder="E.g. VUSA, CSPX, EQQQ, VWRL, AGGH, VFEM, VHYL")
    search_input = search_input.upper() if search_input is not None else None
    search_button = st.form_submit_button("Search")

if search_button:
    reset_query_params(search_input)
    st.rerun()

result_quotes_df = search_symbol(search_input)

first_result = result_quotes_df.iloc[0] if result_quotes_df is not None and result_quotes_df.shape[0] > 0 else None

symbol_name = first_result.name if first_result is not None else None


st.write("##### Search results")

if result_quotes_df is not None:

    st.dataframe(result_quotes_df)
else:
    if search_input is None:
        st.info("Please enter a symbol to search.")
    else:
        st.warning(f"No results found for '{search_input}'.")


if symbol_name is not None:

    st.subheader(f"Symbol: {symbol_name}")

    asset_info = fetch_asset_info(symbol_name)

    combo_asset_info = {**asset_info, **first_result}
    with st.expander("Raw asset info (JSON)"):
        st.json(combo_asset_info)

    st.dataframe(create_asset_info_df(combo_asset_info),
                 hide_index=True,
                 use_container_width=True)

    full_asset_history = fetch_asset_history(symbol_name)

    if full_asset_history is None:
        st.error(f"No historical data found for {symbol_name}.")
        st.stop()

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
        # color=alt.condition(
        #     alt.datum.annual_base_return > 0,
        #     alt.value("#14B3EB"),
        #     alt.value("#EB4C14")
        # )
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
