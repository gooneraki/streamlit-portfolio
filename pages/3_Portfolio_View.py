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


def display_portfolio_weights(result):
    """Enhanced function to display portfolio weights with metrics and performance"""
    portfolio_weights = result.weights

    # Create a DataFrame for better display
    weights_display_df = pd.DataFrame({
        'Asset': portfolio_weights.index,
        'Weight': portfolio_weights.values,
        'Weight %': (portfolio_weights.values * 100)
    }).sort_values('Weight %', ascending=False)

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
        st.metric("Log Sharpe Ratio",
                  f"{result.ann_log_sharpe_ratio:.3f}")

    # Display weights and diversification metrics
    weight_col1, weight_col2 = st.columns([2, 1])
    with weight_col1:
        st.markdown("**Portfolio Weights**")
        st.dataframe(
            weights_display_df,
            column_config={
                "Asset": st.column_config.TextColumn("Asset"),
                "Weight": st.column_config.NumberColumn(
                    "Weight", format="%.4f"
                ),
                "Weight %": st.column_config.NumberColumn(
                    "Weight %", format="%.2f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )

    with weight_col2:
        st.markdown("**Portfolio Statistics**")
        # Show summary statistics
        st.metric("Number of Assets", len(portfolio_weights))
        st.metric("Max Weight", f"{portfolio_weights.max():.2%}")
        st.metric("Min Weight", f"{portfolio_weights.min():.2%}")
        st.metric("Weight Range",
                  f"{portfolio_weights.max() - portfolio_weights.min():.2%}")

        # Show diversification metrics
        portfolio_herfindahl = (portfolio_weights ** 2).sum()
        portfolio_effective_n = 1 / portfolio_herfindahl
        st.metric("Herfindahl Index", f"{portfolio_herfindahl:.4f}")
        st.metric("Effective N", f"{portfolio_effective_n:.1f}")

        if portfolio_effective_n < 3:
            st.warning("⚠️ Low diversification (Effective N < 3)")
        elif portfolio_effective_n < 5:
            st.info("📊 Moderate diversification")
        else:
            st.success("✅ Good diversification")


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
print(f"\nassets_snapshot:\n{assets_snapshot.T}")

total_metrics = assets_snapshot.loc['TOTAL']

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

st.markdown("#### Current Portfolio Composition")
st.dataframe(
    pd.DataFrame(portfolio.get_assets_snapshot().drop(
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

# --- Optimal Portfolio Weights ---
st.markdown("---")
st.markdown("#### 🎯 Portfolio Optimization Strategies")

optimisation_results = portfolio.get_optimisation_results()
if optimisation_results:
    # Create tabs dynamically based on available optimization results
    tab_names = ["📊 Strategy Comparison"]

    # Add tabs for each available optimization strategy
    for strategy_key, strategy_result in optimisation_results.items():
        tab_names.append(f"{strategy_result.emoji} {strategy_result.name}")

    # Always add current portfolio tab at the end
    tab_names.append("⚖️ Current Portfolio")

    strategy_tabs = st.tabs(tab_names)

    with strategy_tabs[0]:
        st.markdown("##### 📈 Portfolio Strategy Performance Comparison")

        # Create comprehensive comparison DataFrame
        comparison_data = []
        for strategy_key, optimal_result in optimisation_results.items():
            if hasattr(optimal_result, 'weights') and isinstance(optimal_result.weights, pd.Series):
                strategy_weights = optimal_result.weights

                # Calculate diversification metrics
                strategy_herfindahl = (strategy_weights ** 2).sum()
                strategy_effective_n = 1 / strategy_herfindahl
                strategy_max_weight = strategy_weights.max()
                strategy_min_weight = strategy_weights.min()
                strategy_weight_range = strategy_max_weight - strategy_min_weight

                comparison_data.append({
                    'Strategy': f"{optimal_result.emoji} {optimal_result.name}",
                    'Annual Return': f"{optimal_result.ann_arith_return:.2%}",
                    'Annual Volatility': f"{optimal_result.ann_vol:.2%}",
                    'Sharpe Ratio': f"{optimal_result.ann_arith_sharpe_ratio:.3f}",
                    'Max Weight': f"{strategy_max_weight:.1%}",
                    'Effective N': f"{strategy_effective_n:.1f}",
                    'Diversification': "Low"
                    if strategy_effective_n < 3
                    else "Moderate"
                    if strategy_effective_n < 5
                    else "Good"
                })

        # Add current portfolio to comparison
        current_weights = assets_snapshot.drop('TOTAL')['latest_weights']
        current_herfindahl = (current_weights ** 2).sum()
        current_effective_n = 1 / current_herfindahl
        current_max_weight = current_weights.max()

        # Calculate current portfolio return (convert CAGR to annual arithmetic return)
        current_cagr = total_metrics['cagr']

        comparison_data.append({
            'Strategy': "💼 Current Portfolio",
            'Annual Return': f"{current_cagr:.2%}",
            'Annual Volatility': "N/A",  # We don't have volatility for current portfolio
            'Sharpe Ratio': "N/A",      # We don't have Sharpe ratio for current portfolio
            'Max Weight': f"{current_max_weight:.1%}",
            'Effective N': f"{current_effective_n:.1f}",
            'Diversification': "Low"
            if current_effective_n < 3
            else "Moderate"
            if current_effective_n < 5
            else "Good"
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
                    "Max Weight": st.column_config.TextColumn("Max Weight"),
                    "Effective N": st.column_config.TextColumn("Effective N"),
                    "Diversification": st.column_config.TextColumn("Diversification")
                },
                hide_index=True,
                use_container_width=True
            )

    # Dynamic strategy tabs - start from index 1 (after comparison tab)
    tab_index = 1
    for strategy_key, strategy_result in optimisation_results.items():
        with strategy_tabs[tab_index]:
            display_portfolio_weights(strategy_result)
        tab_index += 1

    # Current portfolio tab (always last)
    with strategy_tabs[-1]:
        st.markdown("##### 📋 Current Portfolio Analysis")

        # Show current portfolio as comparison
        current_weights = assets_snapshot.drop('TOTAL')['latest_weights']
        current_weights_df = pd.DataFrame({
            'Asset': current_weights.index,
            'Weight': current_weights.values,
            'Weight %': (current_weights * 100).values
        }).sort_values('Weight %', ascending=False)

        current_col1, current_col2 = st.columns([2, 1])
        with current_col1:
            st.markdown("**Current Portfolio Weights**")
            st.dataframe(
                current_weights_df,
                column_config={
                    "Asset": st.column_config.TextColumn("Asset"),
                    "Weight": st.column_config.NumberColumn("Weight", format="%.4f"),
                    "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True
            )

        with current_col2:
            st.markdown("**Current Portfolio Statistics**")
            st.metric("Number of Assets", len(current_weights))
            st.metric("Max Weight", f"{current_weights.max():.2%}")
            st.metric("Min Weight", f"{current_weights.min():.2%}")

            current_herfindahl = (current_weights ** 2).sum()
            current_effective_n = 1 / current_herfindahl
            st.metric("Effective N", f"{current_effective_n:.1f}")

            if current_effective_n < 3:
                st.warning("⚠️ Low diversification")
            elif current_effective_n < 5:
                st.info("📊 Moderate diversification")
            else:
                st.success("✅ Good diversification")

        # Show current portfolio performance
        st.markdown("**Current Portfolio Performance**")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("CAGR", f"{total_metrics['cagr_pct']:.2f}%")
        with perf_col2:
            st.metric("Latest Value", f"{total_metrics['latest_value']:,.0f}")
        with perf_col3:
            st.metric("Trend Deviation",
                      f"{total_metrics['trend_deviation']*100:.2f}%")

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

selected_metric = st.selectbox('Select Metric/Column', all_metrics, index=all_metrics.index(
    'translated_values') if 'translated_values' in all_metrics else 0)

# Prepare data: exclude TOTAL for asset comparison
asset_names = [a for a in portfolio.timeseries_data.columns.get_level_values(
    'Ticker').unique() if a != 'TOTAL']

# Use the new chart function
if selected_metric:
    title = f"All Assets - {selected_metric.replace('_', ' ').title()} Trend"
    fig = display_multi_asset_metric_trend(
        portfolio.timeseries_data, asset_names, selected_metric, title=title)
    st.plotly_chart(fig, use_container_width=True)
