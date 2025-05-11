# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""

from typing import List
import datetime
import streamlit as st
import streamlit.components.v1 as components
from utilities.utilities import AssetDetails,  create_asset_info_df, \
    generate_asset_base_value, append_fitted_data, get_trend_info, \
    get_annual_returns_trend_info, get_quotes_by_symbol, get_history_options
from utilities.constants import BASE_CURRENCY_OPTIONS
from utilities.go_charts import display_trend_go_chart, display_daily_annual_returns_chart
from utilities.app_yfinance import sector_yf, ticker_yf_history, fetch_fx_rate_history, yf_ticket_info

print(f"\n--- Now: {datetime.datetime.now()} ---\n")


def reset_query_params(p_search_input: str):
    """Reset the query parameters."""
    if p_search_input is None or len(p_search_input.strip()) == 0:
        st.query_params.pop("symbol", None)
    else:
        st.query_params.update({"symbol": p_search_input.strip().upper()})


st.title("Symbol Information")

st.write("### Search symbols")

with st.form(key="search_form"):
    search_input = st.text_input("First result will be analyzed",
                                 value=st.query_params.get("symbol"),
                                 max_chars=20,
                                 key="search_input",
                                 placeholder="E.g. VUSA, CSPX, EQQQ, VWRL, AGGH, VFEM, VHYL")

    search_button = st.form_submit_button("Search")

if search_button:
    reset_query_params(search_input)
    st.rerun()


result_quotes_df = get_quotes_by_symbol(search_input)

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

if symbol_name is None:
    st.stop()

st.divider()


tab1, tab2 = st.tabs(["Symbol Information", "Industry Information"])

asset_info = yf_ticket_info(symbol_name)

combo_asset_info = {**first_result, **asset_info}

with tab1:

    st.subheader(f"Symbol: {symbol_name}")

    components.html(
        """
        <style>
            #copy-btn {
                background-color: #f0fcec;
                color: #177233;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                border: 1px solid #e0f5e7;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 2px 6px rgba(33, 195, 84, 0.1);
            }

            #copy-btn:hover {
                background-color: #f0fcec;
                border-color: #21c354;
            }

            #copy-btn:active {
                transform: scale(0.97);
            }
        </style>

        <script>
            function copyURL() {
                navigator.clipboard.writeText(window.parent.location.href)
                    .then(() => {
                        const btn = document.getElementById("copy-btn");
                        btn.innerText = "✅ Copied!";
                        setTimeout(() => btn.innerText = "📋 Copy current URL", 2000);
                    })
                    .catch(err => alert("Failed to copy URL: " + err));
            }
        </script>

        <button id="copy-btn" onclick="copyURL()">📋 Copy current URL</button>
        """,
        height=50)

    with st.expander("Raw asset info (JSON)"):
        st.json(combo_asset_info)

    # st.dataframe(create_asset_info_df(combo_asset_info),
    #              hide_index=True,
    #              use_container_width=True)
    for label, value in create_asset_info_df(combo_asset_info):
        st.markdown(f"**{label}**: {value}")

    full_asset_history = ticker_yf_history(symbol_name)

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

    history_options = get_history_options(
        full_asset_base_history['base_value'].shape[0])

    history_options_keys = list(history_options.keys())

    with col2:
        selected_period_key = st.selectbox(
            "Select period", history_options_keys, index=history_options_keys.index("10 Years") if
            "10 Years" in history_options_keys else 0)

    selected_period_value = history_options[selected_period_key]

    # Generate the data for the selected period
    periodic_asset_history_with_fit, cagr, cagr_fitted, base_over_under = append_fitted_data(
        full_asset_base_history, selected_period_value)

    # Display the CAGR and CAGR for the fitted data

    col1, col2 = st.columns([1, 2])

    with col1:
        trend_info_df = get_trend_info(periodic_asset_history_with_fit)
        st.dataframe(trend_info_df, hide_index=True)
        st.write("*CAGR: Compound Annual Growth Rate*")

    # Create the line chart
    with col2:
        value_fig = display_trend_go_chart(
            periodic_asset_history_with_fit,  title_name=f"{symbol_name} - {base_currency} - {selected_period_key}")
        if value_fig is None:
            st.warning("No valid data to plot.")
        else:
            st.plotly_chart(value_fig, config={
                            "displayModeBar": False}, use_container_width=True)

    st.markdown("#### Daily Annual Returns")

    annual_returns_info, mean = get_annual_returns_trend_info(
        periodic_asset_history_with_fit)

    annual_col1, annual_col2 = st.columns([1, 2])

    with annual_col1:
        st.dataframe(annual_returns_info, hide_index=True)

    with annual_col2:
        annual_returns_fig = display_daily_annual_returns_chart(
            df=periodic_asset_history_with_fit,
            annual_column='annual_base_return',
            mean_value=mean
        )
        if annual_returns_fig is None:
            st.warning("No valid data to plot.")
        else:
            st.plotly_chart(annual_returns_fig, use_container_width=True, config={
                            "displayModeBar": False})

with tab2:

    symbol_sector = combo_asset_info.get("sector", None)
    if symbol_sector is None:
        st.info(f"Symbol '{symbol_name}' does not have a sector.")
        st.stop()
    else:
        st.write(f"### Sector: {symbol_sector}")
        symbol_sector_key = combo_asset_info.get(
            "sectorKey", None)

        sector_data = sector_yf(
            symbol_sector_key)

        if sector_data.get('overview') is None:
            st.warning(f"No sector overview found. '{symbol_sector_key}'")
        else:
            st.write("#### Sector Overview")
            st.write(sector_data.get('overview'))

            if sector_data.get('top_companies') is None:
                st.warning(f"No top companies found. '{symbol_sector_key}'")
            else:
                st.write("#### Top Companies")
                st.dataframe(sector_data.get('top_companies'))

            if sector_data.get('top_etfs') is None:
                st.warning(f"No top ETFs found. '{symbol_sector_key}'")
            else:
                st.write("#### Top ETFs")
                st.dataframe(sector_data.get('top_etfs'))
