# pylint: disable=C0103
"""Get the information of a stock symbol from Yahoo Finance API."""
import datetime
import json
import pandas as pd
import streamlit as st

from utilities.constants import ASSETS_POSITIONS_DEFAULT, BASE_CURRENCY_OPTIONS
from utilities.go_charts import display_trend_go_chart_2, display_multi_asset_metric_trend
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

first_date, last_date, \
    number_of_points, number_of_days, number_of_years, \
    points_per_year, points_per_month = portfolio.get_period_info()

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
    f"📈 **Analysis Coverage:** {number_of_days:,} days ({number_of_years:.1f} years) "
    f"with {number_of_points:,} data points (avg {points_per_year:.0f} points/year)"
)


st.markdown("---")


# --- Portfolio TOTAL summary metrics ---
assets_metrics = portfolio.get_assets_metrics()
total_metrics = assets_metrics.loc['TOTAL']

st.markdown("#### 🏆 Portfolio Total Summary")
sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
with sum_col1:
    st.metric("CAGR", f"{total_metrics['cagr_pct']:.2f}%")
with sum_col2:
    st.metric("CAGR Fitted", f"{total_metrics['cagr_fitted_pct']:.2f}%")
with sum_col3:
    st.metric("Latest Value", f"{total_metrics['latest_value']:,.0f}")
with sum_col4:
    st.metric("Trend Deviation %",
              f"{total_metrics['trend_deviation']*100:.2f}%")
with sum_col5:
    st.metric("Trend Dev. Z-Score",
              f"{total_metrics['trend_deviation_z_score']:.2f}")
st.caption(f"All values as of {last_date.strftime('%Y-%m-%d')}")

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
        "trend_deviation": st.column_config.NumberColumn(
            label="Trend Deviation",
            format="percent"
        )
    })

st.caption(f"All values as of {last_date.strftime('%Y-%m-%d')}")

st.markdown("---")

st.markdown("#### Portfolio Performance")

# Get available assets from timeseries data
available_assets = portfolio.timeseries_data.columns.get_level_values(
    'Ticker').unique().tolist()

# Create columns for asset selection and time period filter
col1, col2 = st.columns([1, 1])

with col1:
    # Asset selection
    selected_asset = st.selectbox(
        "Select Asset",
        available_assets,
        index=available_assets.index(
            "TOTAL") if "TOTAL" in available_assets else 0
    )

with col2:
    # Time period filter - check available data range
    total_years = number_of_years
    year_options = []

    if total_years >= 1:
        year_options.append("1 Year")
    if total_years >= 3:
        year_options.append("3 Years")
    if total_years >= 5:
        year_options.append("5 Years")
    if total_years >= 10:
        year_options.append("10 Years")
    if total_years > 10:
        year_options.append("10+ Years (All)")

    # Default to the longest available period
    default_period = year_options[-1] if year_options else "1 Year"

    selected_period = st.selectbox(
        "Display Period",
        year_options
    )

# Get data for selected asset
asset_data = portfolio.timeseries_data.xs(
    selected_asset, level='Ticker', axis=1)

# Filter data based on selected period
if selected_period == "1 Year":
    asset_data = asset_data.tail(round(points_per_year))
elif selected_period == "3 Years":
    asset_data = asset_data.tail(round(points_per_year * 3))
elif selected_period == "5 Years":
    asset_data = asset_data.tail(round(points_per_year * 5))
elif selected_period == "10 Years":
    asset_data = asset_data.tail(round(points_per_year * 10))
# For "10+ Years (All)", use all data (no filtering)

# Create the chart
asset_fig = display_trend_go_chart_2(
    df=asset_data,
    value_column='translated_values',
    fitted_column='translated_fitted_values',
    secondary_column='trend_deviation_z_score',
    title_name=f"{selected_asset} - Performance Analysis"
)

if asset_fig is None:
    st.warning("No valid data to plot.")
else:
    st.plotly_chart(asset_fig, use_container_width=True)

# --- Reverse chart: select metric, plot all assets ---
st.markdown('---')
st.markdown('#### Compare All Assets by Metric')

# Get available metrics (excluding weights, as those are proportions)
all_metrics = [m for m in portfolio.timeseries_data.columns.get_level_values('Metric').unique()
               if m not in ['weights']]

selected_metric = st.selectbox('Select Metric/Column', all_metrics, index=all_metrics.index(
    'translated_values') if 'translated_values' in all_metrics else 0)

# Prepare data: exclude TOTAL for asset comparison
asset_names = [a for a in portfolio.timeseries_data.columns.get_level_values(
    'Ticker').unique() if a != 'TOTAL']

# Use the new chart function
title = f"All Assets - {selected_metric.replace('_', ' ').title()} Trend"
fig = display_multi_asset_metric_trend(
    portfolio.timeseries_data, asset_names, selected_metric, title=title)
st.plotly_chart(fig, use_container_width=True)
