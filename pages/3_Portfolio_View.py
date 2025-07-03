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

    # Display authentication status
    try:
        if username == st.secrets["DB_USERNAME"]:
            st.success("✅ Authenticated - Loading personalized portfolio data")
        else:
            st.warning(
                "⚠️ Using default portfolio data. Enter your username to load personalized portfolio data.")
    except KeyError:
        st.warning("⚠️ Using default portfolio data")


def display_portfolio_weights(result, p_assets_snapshot=None):
    """Enhanced function to display portfolio weights with metrics and performance"""
    portfolio_weights = result.weights

    # Create a DataFrame for better display with currency information
    weights_display_df = pd.DataFrame({
        'Asset': portfolio_weights.index,
        'Weight %': (portfolio_weights.values * 100)
    }).sort_values('Weight %', ascending=False)

    # Add currency information if available
    if p_assets_snapshot is not None:
        # Get currency information for each asset
        currency_info = {}
        for asset in portfolio_weights.index:
            if asset in p_assets_snapshot.index:
                currency_info[asset] = p_assets_snapshot.loc[asset, 'currency']

        weights_display_df['Currency'] = weights_display_df['Asset'].map(
            currency_info.get)

    # Display performance metrics at the top
    st.markdown(
        f"##### {result.emoji} {result.name} Performance")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Annual Return", f"{result.ann_arith_return:.2%}")
    with metric_col2:
        st.metric("Annual Volatility", f"{result.ann_vol:.2%}")
    with metric_col3:
        st.metric("Sharpe Ratio",
                  f"{result.ann_arith_sharpe_ratio:.3f}")
    with metric_col4:
        st.metric("Herfindahl Index",
                  f"{result.herfindahl_index:.3f}")

    st.markdown("**Portfolio Weights**")
    column_config = {
        "Asset": st.column_config.TextColumn("Asset"),
        "Weight %": st.column_config.NumberColumn(
            "Weight %", format="%.2f%%"
        )
    }
    if 'Currency' in weights_display_df.columns:
        column_config["Currency"] = st.column_config.TextColumn("Currency")

    st.dataframe(
        weights_display_df,
        column_config=column_config,
        hide_index=True,
        use_container_width=True
    )


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

# Get period info
period_info = portfolio.get_period_info()

last_date = period_info.last_date
points_per_year = period_info.points_per_year

# Display sampling period information
st.markdown("#### 📊 Data Sampling Period")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Start Date", period_info.first_date.strftime('%Y-%m-%d'))
with col2:
    st.metric("End Date", last_date.strftime('%Y-%m-%d'))
with col3:
    st.metric("Sample Period", f"{period_info.number_of_years:.1f} years")
with col4:
    st.metric("Data Points", f"{period_info.number_of_points:,}")

st.info(
    f"📈 **Analysis Coverage:** {period_info.number_of_days:,} days ({period_info.number_of_years:.1f} years) "
    f"with {period_info.number_of_points:,} data points (avg {points_per_year:.0f} points/year)"
)


st.markdown("---")


# --- Portfolio TOTAL summary metrics ---
assets_snapshot = portfolio.get_assets_snapshot()


total_metrics = assets_snapshot.loc['TOTAL']

st.markdown("#### 🏆 Portfolio Total Summary")
sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
with sum_col1:
    st.metric("Rolling 1M Return",
              f"{total_metrics['rolling_1m_return_pct']:.1f}%")
    st.caption(f"Z-Score: {total_metrics['rolling_1m_return_z_score']:.2f}")

with sum_col2:
    st.metric("Rolling 1Q Return",
              f"{total_metrics['rolling_1q_return_pct']:.1f}%")
    st.caption(f"Z-Score: {total_metrics['rolling_1q_return_z_score']:.2f}")

with sum_col3:
    st.metric("Rolling 1Y Return",
              f"{total_metrics['rolling_1y_return_pct']:.1f}%")
    st.caption(f"Z-Score: {total_metrics['rolling_1y_return_z_score']:.2f}")

with sum_col4:
    st.metric("CAGR", f"{total_metrics['cagr_pct']:.1f}%")
    st.caption(f"Fitted: {total_metrics['cagr_fitted_pct']:.1f}%")

with sum_col5:
    st.metric("Portfolio Value", f"{total_metrics['translated_values']:,.0f}")
    st.caption(
        f"Z-Score: {total_metrics['trend_deviation_z_score']:.2f}")

st.caption(f"All values as of {last_date.strftime('%Y-%m-%d')}")

st.markdown("---")

st.markdown("#### Current Portfolio Composition")

# Select and reorder columns for better display
display_columns = [
    'currency', 'position', 'translated_close', 'translated_values', 'weights_pct',

    'cagr_pct', 'rolling_1y_return_pct', 'rolling_1q_return_pct', 'rolling_1m_return_pct',
    'cagr_fitted_pct',
    'trend_deviation_z_score', 'rolling_1y_return_z_score', 'rolling_1q_return_z_score', 'rolling_1m_return_z_score']

# Get the assets snapshot and select only the columns we want to display
assets_display = portfolio.get_assets_snapshot()[display_columns]

st.dataframe(
    assets_display,
    column_config={
        "position": st.column_config.NumberColumn(
            label="Position", format="localized"),
        "weights_pct": st.column_config.NumberColumn(
            label="Weight", format="%.1f%%"),
        "translated_close": st.column_config.NumberColumn(
            label="Close Price", format="%.2f"),
        "translated_values": st.column_config.NumberColumn(
            label="Value", format="accounting"),
        "trend_deviation_z_score": st.column_config.NumberColumn(
            label="Value Trend Z", format="%.2f"),

        "cagr_pct": st.column_config.NumberColumn(
            label="CAGR", format="%.1f%%"),
        "cagr_fitted_pct": st.column_config.NumberColumn(
            label="CAGR Fitted", format="%.1f%%"),

        "rolling_1m_return_pct": st.column_config.NumberColumn(
            label="1M Return", format="%.1f%%"),
        "rolling_1q_return_pct": st.column_config.NumberColumn(
            label="1Q Return", format="%.1f%%"),
        "rolling_1y_return_pct": st.column_config.NumberColumn(
            label="1Y Return", format="%.1f%%"),

        "rolling_1m_return_z_score": st.column_config.NumberColumn(
            label="1M Return Z", format="%.2f"),
        "rolling_1q_return_z_score": st.column_config.NumberColumn(
            label="1Q Return Z", format="%.2f"),
        "rolling_1y_return_z_score": st.column_config.NumberColumn(
            label="1Y Return Z", format="%.2f"),

        "currency": st.column_config.TextColumn(label="Currency")
    })

st.caption(f"All values as of {last_date.strftime('%Y-%m-%d')}")

# --- Optimal Portfolio Weights ---
st.markdown("---")
st.markdown("#### 🎯 Portfolio Optimization Strategies")

optimisation_results = portfolio.get_optimisation_results()
if optimisation_results:
    # Create tabs dynamically based on available optimization results
    tab_names = ["📊 Strategy Comparison"]

    # Add tabs for each available optimization strategy
    for strategy_result in optimisation_results:
        tab_names.append(f"{strategy_result.emoji} {strategy_result.name}")

    strategy_tabs = st.tabs(tab_names)

    with strategy_tabs[0]:
        st.markdown("##### 📈 Portfolio Strategy Performance Comparison")

        # Create comprehensive comparison DataFrame
        comparison_data = []
        for optimal_result in optimisation_results:
            strategy_weights = optimal_result.weights

            # Calculate diversification metrics
            strategy_herfindahl = (strategy_weights ** 2).sum()

            strategy_max_weight = strategy_weights.max()
            strategy_min_weight = strategy_weights.min()

            comparison_data.append({
                'Strategy': f"{optimal_result.emoji} {optimal_result.name}",
                'Annual Return': f"{optimal_result.ann_arith_return:.1%}",
                'Annual Volatility': f"{optimal_result.ann_vol:.1%}",
                'Sharpe Ratio': f"{optimal_result.ann_arith_sharpe_ratio:.3f}",
                'Min Weight': f"{strategy_min_weight:.1%}",
                'Max Weight': f"{strategy_max_weight:.1%}",
                'Herfindahl Index': f"{strategy_herfindahl:.3f}"
            })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df,
                column_config={
                    "Strategy": st.column_config.TextColumn("Strategy"),
                    "Annual Return": st.column_config.TextColumn("Annual Return"),
                    "Annual Volatility": st.column_config.TextColumn("Annual Volatility"),
                    "Sharpe Ratio": st.column_config.TextColumn("Sharpe Ratio"),
                    "Min Weight": st.column_config.TextColumn("Min Weight"),
                    "Max Weight": st.column_config.TextColumn("Max Weight"),
                    "Herfindahl Index": st.column_config.TextColumn("Herfindahl Index"),
                    "Effective N": st.column_config.TextColumn("Effective N"),
                    "Diversification": st.column_config.TextColumn("Diversification")
                },
                hide_index=True,
                use_container_width=True
            )

    # Dynamic strategy tabs - start from index 1 (after comparison tab)
    tab_index = 1
    for strategy_result in optimisation_results:
        with strategy_tabs[tab_index]:
            display_portfolio_weights(strategy_result, assets_snapshot)
        tab_index += 1

else:
    st.info(
        "Portfolio optimization not available - insufficient data or optimization failed.")


st.markdown("---")

st.markdown("#### Portfolio Performance")

# Get available assets from timeseries data
available_assets = portfolio.timeseries_data.columns.get_level_values(
    'Ticker').unique().tolist()

# Create columns for asset selection and time period filter
asset_col1, asset_col2 = st.columns([1, 1])

with asset_col1:
    # Asset selection
    selected_asset = st.selectbox(
        "Select Asset",
        available_assets,
        index=available_assets.index(
            "TOTAL") if "TOTAL" in available_assets else 0
    )

with asset_col2:
    # Time period filter - check available data range
    total_years = period_info.number_of_years
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

# Create columns for metric selection and time period filter
metric_select_col1, metric_select_col2 = st.columns([1, 1])

with metric_select_col1:
    selected_metric = st.selectbox('Select Metric/Column', all_metrics, index=all_metrics.index(
        'translated_values') if 'translated_values' in all_metrics else 0)

with metric_select_col2:
    # Time period filter - reuse the same logic as single asset view
    total_years = period_info.number_of_years
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

    selected_period_multi = st.selectbox(
        "Display Period",
        year_options,
        key="multi_asset_period"
    )

# Prepare data: exclude TOTAL for asset comparison
asset_names = [a for a in portfolio.timeseries_data.columns.get_level_values(
    'Ticker').unique() if a != 'TOTAL']

# Use the new chart function
if selected_metric:
    # Filter data based on selected period
    filtered_data = portfolio.timeseries_data.copy()

    if selected_period_multi == "1 Year":
        filtered_data = filtered_data.tail(round(points_per_year))
    elif selected_period_multi == "3 Years":
        filtered_data = filtered_data.tail(round(points_per_year * 3))
    elif selected_period_multi == "5 Years":
        filtered_data = filtered_data.tail(round(points_per_year * 5))
    elif selected_period_multi == "10 Years":
        filtered_data = filtered_data.tail(round(points_per_year * 10))
    # For "10+ Years (All)", use all data (no filtering)

    title = f"All Assets - {selected_metric.replace('_', ' ').title()} Trend ({selected_period_multi})"
    fig = display_multi_asset_metric_trend(
        filtered_data, asset_names, selected_metric, title=title)
    st.plotly_chart(fig, use_container_width=True)
