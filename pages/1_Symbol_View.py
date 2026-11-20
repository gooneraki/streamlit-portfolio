
"""Get the information of a stock symbol from Yahoo Finance API."""

import logging
from statistics import NormalDist

import pandas as pd
import streamlit as st
from utilities.utilities import create_asset_info_df, \
    generate_asset_base_value, append_fitted_data, get_period_returns, \
    get_quotes_by_symbol, get_history_options, get_period_info, \
    fetch_fx_rate_history, ticker_yf_history
from utilities.constants import BASE_CURRENCY_OPTIONS
from utilities.go_charts import display_trend_go_chart, display_daily_annual_returns_chart
from utilities.app_yfinance import sector_yf,   yf_ticket_info

logger = logging.getLogger(__name__)
logger.debug("Symbol View loaded")

NORMAL_Z_BANDS = tuple(NormalDist().inv_cdf(p) for p in (0.75, 0.95, 0.995))

st.set_page_config(page_title="Symbol View", layout="wide")


def reset_query_params(p_search_input: str):
    """Reset the query parameters."""
    if p_search_input is None or len(p_search_input.strip()) == 0:
        st.query_params.pop("symbol", None)
    else:
        st.query_params.update({"symbol": p_search_input.strip().upper()})


def describe_return_z_score(z_score: float) -> str:
    """Describe a return relative to its historical average."""
    typical_limit, very_limit, exceptional_limit = NORMAL_Z_BANDS
    if pd.isna(z_score):
        return "N/A"
    if z_score <= -exceptional_limit:
        return "Exceptionally low"
    if z_score <= -very_limit:
        return "Very low"
    if z_score <= -typical_limit:
        return "Low"
    if z_score < typical_limit:
        return "Typical"
    if z_score < very_limit:
        return "High"
    if z_score < exceptional_limit:
        return "Very high"
    return "Exceptionally high"


st.title("Symbol Information")

st.write("### Search symbols")

with st.form(key="search_form"):
    search_input = st.text_input("First result will be analyzed",
                                 value=st.query_params.get("symbol"),
                                 max_chars=20,
                                 key="search_input",
                                 placeholder="E.g. VUSA, VWCE, CSPX, EQQQ, VWRL, AGGH, VFEM, VHYL")

    search_button = st.form_submit_button("Search")

if search_button:
    reset_query_params(search_input)
    st.rerun()


result_quotes_df = get_quotes_by_symbol(search_input)

first_result = result_quotes_df.iloc[0] if result_quotes_df is not None and result_quotes_df.shape[0] > 0 else None

symbol_name = first_result.name if first_result is not None else None

st.write("##### Search results")

if result_quotes_df is not None:

    st.dataframe(result_quotes_df, width="stretch")
else:
    if search_input is None:
        st.info("Please enter a symbol to search.")
    else:
        st.warning(f"No results found for '{search_input}'.")

if symbol_name is None:
    st.stop()

st.divider()


tab1, tab2, tab3 = st.tabs(
    ["Metrics", "Symbol Information", "Industry Information"])

asset_info_result = yf_ticket_info(symbol_name)
if asset_info_result.error is not None:
    st.error(f"Unable to fetch symbol information: {asset_info_result.error}")
    st.stop()

combo_asset_info = {**first_result, **asset_info_result.data}

with tab2:

    st.subheader(f"Symbol: {symbol_name}")

    st.iframe(
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

    asset_info_rows = create_asset_info_df(combo_asset_info)
    key_statistic_labels = {
        "Current Price",
        "Market Cap",
        "Revenue Growth (YoY)",
        "Earnings Growth (YoY)",
        "Profit Margin",
        "Forward P/E",
        "Dividend Yield",
        "Price Growth (1 Year)",
        "52-Week Range",
        "52-Week Range Position",
        "52-Week Range Width",
        "Price vs 200-Day Average",
        "Years Since First Trade",
        "Fund Assets",
        "Average Daily Volume",
    }
    business_summary = next(
        (value for label, value in asset_info_rows if label == "Business Summary"), None)

    overview_rows = [
        [label, value]
        for label, value in asset_info_rows
        if label not in key_statistic_labels | {"Business Summary"}
    ]
    key_statistic_rows = [
        [label, value]
        for label, value in asset_info_rows
        if label in key_statistic_labels
    ]

    overview_column, statistics_column = st.columns(2)
    with overview_column:
        if overview_rows:
            st.markdown("#### Overview")
            st.table(
                [{"Label": label, "Value": value}
                    for label, value in overview_rows],
                hide_index=True,
                width="content",
            )

    with statistics_column:
        if key_statistic_rows:
            st.markdown("#### Key Statistics")
            st.table(
                [{"Label": label, "Value": value}
                    for label, value in key_statistic_rows],
                hide_index=True,
                width="content",
            )

    if business_summary:
        st.markdown("#### Business Summary")
        st.write(business_summary)

with tab1:

    asset_name = asset_info_result.data.get(
        "longName") or asset_info_result.data.get("shortName")
    st.subheader(
        f"{asset_name} ({symbol_name})" if asset_name else str(symbol_name))

    full_asset_history = ticker_yf_history(symbol_name)

    if full_asset_history is None:
        st.error(f"No historical data found for {symbol_name}.")
        st.stop()

    currency_col, _ = st.columns(2)

    with currency_col:
        base_currency = st.selectbox(
            "Select currency", BASE_CURRENCY_OPTIONS, key="currency_input")

    # Fetch the fx rate history for the asset currency
    full_fx_rate_history = fetch_fx_rate_history(
        asset_info_result.data['currency'], base_currency)
    if not isinstance(full_fx_rate_history, pd.Series):
        st.error(f"Unable to fetch currency history: {full_fx_rate_history}")
        st.stop()

    # Add the fx rate history to the asset history
    full_asset_base_history = generate_asset_base_value(
        full_asset_history, full_fx_rate_history)

    if len(full_asset_base_history) < 2:
        st.warning(
            "At least two days of price and currency history are needed for metrics.")
        st.stop()
    if (full_asset_base_history["base_value"] <= 0).any():
        st.error("Historical values must be positive to calculate the growth trend.")
        st.stop()

    number_of_points, number_of_days, number_of_years, points_per_year, _ = get_period_info(
        full_asset_base_history)
    full_history_with_fit, cagr, cagr_fitted, base_over_under = append_fitted_data(
        full_asset_base_history, len(full_asset_base_history))
    period_returns = get_period_returns(
        full_history_with_fit["base_value"], points_per_year)
    full_history_with_fit["annual_base_return"] = period_returns["1Y"]
    latest_date = full_history_with_fit.index[-1]

    sample_years = (latest_date - full_history_with_fit.index[0]).days / 365.25
    st.markdown("#### Long-term growth")
    st.caption(
        f"Full available history in {base_currency}: "
        f"{full_history_with_fit.index[0]:%Y-%m-%d} to {latest_date:%Y-%m-%d}. "
        f"{number_of_points:,} observations over {number_of_days:,} days "
        f"({points_per_year:.0f} observations per year). "
        "Prices are adjusted closes converted to the selected currency."
    )
    growth_col1, growth_col2, growth_col3 = st.columns(3)
    growth_col1.metric("Sample years", f"{sample_years:.1f}")
    growth_col2.metric(
        "Actual CAGR", f"{cagr:.1%}",
        help="Compound annual growth rate between the first and latest actual prices.")
    growth_col3.metric(
        "Trend CAGR", f"{cagr_fitted:.1%}",
        help="Compound annual growth rate between the first and latest prices on the full-history exponential trend.")
    price_col1, price_col2, price_col3 = st.columns(3)
    price_col1.metric("Latest actual price",
                      f"{full_history_with_fit['base_value'].iloc[-1]:,.2f} {base_currency}")
    price_col2.metric("Latest trend price",
                      f"{full_history_with_fit['fitted'].iloc[-1]:,.2f} {base_currency}")
    price_col3.metric(
        "Above/below trend", f"{base_over_under:+.1%}",
        help="The latest actual price relative to the latest fitted trend price.")

    st.markdown("#### Rolling returns")
    st.caption(
        f"Simple price returns as of {latest_date:%Y-%m-%d} compared with the average "
        "of all overlapping periods in the full history. 1D compares consecutive observations; "
        "longer windows use the average observation frequency and are approximate. "
        "Z-score measures how many population standard deviations the latest return is above or below "
        "its historical average. Dates without both a price and an exchange rate are excluded. "
        "N/A means insufficient history or zero standard deviation."
    )
    historical_average = period_returns.mean()
    historical_std = period_returns.std(ddof=0)
    latest_z_scores = (period_returns.iloc[-1] - historical_average).div(
        historical_std.where(
            (historical_std > 0) & (period_returns.max() != period_returns.min())))
    period_labels = {
        "1D": "1 Day", "1W": "1 Week", "1M": "1 Month",
        "1Q": "1 Quarter", "1Y": "1 Year",
    }
    returns_summary = pd.DataFrame({
        "Period": [period_labels[period] for period in period_returns.columns],
        "Vs. average": [describe_return_z_score(value)
                        for value in latest_z_scores],
        "Current return": [
            f"{value:.{1 if period == '1D' else 1}%}" if pd.notna(
                value) else "N/A"
            for period, value in period_returns.iloc[-1].items()
        ],
        "Historical average": [
            f"{value:.{1 if period == '1D' else 1}%}" if pd.notna(
                value) else "N/A"
            for period, value in historical_average.items()
        ],
        "Z-score": [f"{value:+.2f}" if pd.notna(value) else "N/A"
                    for value in latest_z_scores],
    })
    comparison_styles = {
        "Exceptionally low": "background-color: #ef9a9a; color: #4a1010",
        "Very low": "background-color: #ffcdd2; color: #4a1010",
        "Low": "background-color: #ffebee; color: #4a1010",
        "Typical": "background-color: #e5e7eb; color: #1f2937",
        "High": "background-color: #e8f5e9; color: #103c1c",
        "Very high": "background-color: #c8e6c9; color: #103c1c",
        "Exceptionally high": "background-color: #a5d6a7; color: #103c1c",
        "N/A": "",
    }
    styled_returns = returns_summary.style.map(
        lambda value: comparison_styles[value], subset=["Vs. average"])
    st.dataframe(styled_returns, hide_index=True, width="stretch",
                 alt="Current returns compared with historical averages")
    st.caption(
        f"Vs. average: typical below |Z| approximately {NORMAL_Z_BANDS[0]:.3f}; "
        f"low/high from {NORMAL_Z_BANDS[0]:.3f}; very low/high from {NORMAL_Z_BANDS[1]:.3f}; "
        f"exceptionally low/high from {NORMAL_Z_BANDS[2]:.3f}. "
        "These are normal-distribution reference bands for the central 50%, 90% and 99%; "
        "real returns may not follow that distribution. Labels use unrounded z-scores; "
        "colours compare to average, not whether returns are positive."
    )

    st.markdown("#### Price and trend")
    history_options = get_history_options(
        number_of_points, number_of_years, points_per_year)
    history_options_keys = list(history_options)
    period_col, _ = st.columns(2)
    with period_col:
        selected_period_key = st.selectbox(
            "Display Period", history_options_keys,
            index=history_options_keys.index("1 Year") if "1 Year" in history_options_keys else 0)
    displayed_history = full_history_with_fit.tail(
        history_options[selected_period_key])
    value_fig = display_trend_go_chart(
        displayed_history, title_name=f"{symbol_name} - {base_currency} - {selected_period_key}")
    if value_fig is None:
        st.warning("No valid data to plot.")
    else:
        st.plotly_chart(value_fig, config={"displayModeBar": False}, width="stretch",
                        alt="Actual value and full-history growth trend")

    st.markdown("#### 1-year return history")
    st.caption("The dashed line is the full-history average.")
    annual_returns_fig = display_daily_annual_returns_chart(
        df=displayed_history,
        annual_column="annual_base_return",
        average_value=historical_average["1Y"]
    )
    if annual_returns_fig is None:
        st.warning("Not enough history to display 1-year returns.")
    else:
        st.plotly_chart(annual_returns_fig, width="stretch", config={"displayModeBar": False},
                        alt="Rolling one-year returns and full-history average")

with tab3:

    symbol_sector = combo_asset_info.get("sector", None)
    if symbol_sector is None:
        st.info(f"Symbol '{symbol_name}' does not have a sector.")
        st.stop()
    else:
        st.write(f"### Sector: {symbol_sector}")
        symbol_sector_key = combo_asset_info.get(
            "sectorKey", None)

        sector_data_result = sector_yf(
            symbol_sector_key)

        if sector_data_result.error is not None:
            st.warning(
                f"⚠️ Unable to fetch sector data: {sector_data_result.error}")
        else:
            sector_data = sector_data_result.data

            if sector_data.get('overview') is None:
                st.warning(f"No sector overview found. '{symbol_sector_key}'")
            else:
                st.write("#### Sector Overview")
                st.write(sector_data.get('overview'))

                if sector_data.get('top_companies') is None:
                    st.warning(
                        f"No top companies found. '{symbol_sector_key}'")
                else:
                    st.write("#### Top Companies")
                    st.dataframe(sector_data.get('top_companies'))

                if sector_data.get('top_etfs') is None:
                    st.warning(f"No top ETFs found. '{symbol_sector_key}'")
                else:
                    st.write("#### Top ETFs")
                    st.dataframe(sector_data.get('top_etfs'))
