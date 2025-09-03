# pylint: disable=invalid-name
"""
This page is used to fetch the data for the US market
"""
import time
from typing import Any, Literal
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import yfinance as yf
from utilities.utilities import get_exp_fitted_data, get_rolling_exp_fit
from utilities.go_charts import display_symbol_metrics_chart, display_trend_go_chart_2
from utilities.app_yfinance import tickers_yf
from classes.exp_fit_backtester import ExpFitBacktester

print(f"\n--- US Market Page loaded at {time.strftime('%H:%M:%S')} ---\n")

st.set_page_config(
    page_title="US Market",
    page_icon=":bar_chart:",
    layout="wide")


BENCHMARK_SYMBOL = "SPY"

# Sector ETF symbols for sector-based analysis
SECTOR_SYMBOLS = [
    "XLC",   # Communication Services
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLF",   # Financial
    "XLV",   # Health Care
    "XLI",   # Industrial
    "XLK",   # Technology
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLU",   # Utilities
]

def validate_symbols(symbols):
    """
    Validate symbols by testing them individually and return valid ones.
    Returns tuple of (valid_symbols, invalid_symbols, error_details)
    """
    from utilities.app_yfinance import tickers_yf
    
    valid_symbols = []
    invalid_symbols = []
    error_details = {}
    
    for symbol in symbols:
        try:
            result = tickers_yf([symbol])
            if isinstance(result, dict) and not result['history'].empty:
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
                error_details[symbol] = "No data returned"
        except Exception as e:
            invalid_symbols.append(symbol)
            error_details[symbol] = str(e)
    
    return valid_symbols, invalid_symbols, error_details

# Validate symbols before creating ExpFitBacktester
st.write("**Validating market symbols...**")
with st.spinner("Testing symbol data availability..."):
    valid_symbols, invalid_symbols, error_details = validate_symbols(SECTOR_SYMBOLS)

# Show validation results
if invalid_symbols:
    st.warning(f"⚠️ Some symbols are unavailable and will be excluded: {invalid_symbols}")
    with st.expander("View error details"):
        for symbol, error in error_details.items():
            st.write(f"**{symbol}**: {error}")

if valid_symbols:
    st.success(f"✅ Using {len(valid_symbols)} valid symbols: {valid_symbols}")
    
    # Try to create ExpFitBacktester with valid symbols only
    try:
        exp_fit_backtester = ExpFitBacktester(valid_symbols)
        tickers_history = exp_fit_backtester.get_tickers_history()
        main_dataframes = exp_fit_backtester.get_main_dataframes()
        
        # Continue with the rest of the page
        first_level_columns = [*set(main_dataframes.columns.get_level_values(0))]
        second_level_columns = sorted([*set(main_dataframes.columns.get_level_values(1))])

    except Exception as e:
        # Even with valid symbols, we might have other issues
        st.error(f"Error processing valid symbols: {str(e)}")
        valid_symbols = []  # Force demo mode
        
else:
    st.error("❌ No valid symbols found. All symbols are currently unavailable.")

# Only offer demo mode if no valid symbols are available
if not valid_symbols:
    st.info("This might be due to network connectivity issues or invalid symbols. Please check your internet connection and try again.")
    
    # Add demo mode toggle
    use_demo_mode = st.checkbox("Enable Demo Mode (with mock data)", value=False)
    
    if use_demo_mode:
        st.warning("Using demo mode with mock data for testing purposes.")
        
        # Create mock data for demonstration
        import numpy as np
        
        # Generate mock price data
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        symbols = ['XLC', 'XLY', 'XLP', 'XLE', 'TOTAL']
        
        # Create realistic mock price data with trends
        np.random.seed(42)  # For reproducible demo data
        mock_prices = {}
        
        for i, symbol in enumerate(symbols):
            # Different starting prices and growth patterns
            base_price = 100 + i * 50
            growth_rate = 0.0002 + i * 0.0001  # Different growth rates
            volatility = 0.02 + i * 0.005  # Different volatilities
            
            # Generate price series with trend and noise
            trend = np.exp(growth_rate * np.arange(len(dates)))
            noise = np.random.normal(0, volatility, len(dates))
            prices = base_price * trend * (1 + noise)
            mock_prices[symbol] = prices
        
        # Create mock tickers_history
        tickers_history = pd.DataFrame(mock_prices, index=dates)
        tickers_history = tickers_history.drop('TOTAL', axis=1)  # Remove TOTAL for now
        
        # Create mock main_dataframes structure similar to ExpFitBacktester output
        price_data = tickers_history.copy()
        
        # Add TOTAL column
        price_data['TOTAL'] = price_data.mean(axis=1)
        
        # Create benchmark positions (equal weight)
        n_symbols = len(price_data.columns)
        initial_capital = 1000
        equal_weight = 1.0 / (n_symbols - 1)  # Exclude TOTAL from weight calculation
        
        b_position_data = pd.DataFrame(index=price_data.index, columns=price_data.columns)
        for symbol in price_data.columns:
            if symbol != 'TOTAL':
                initial_price = price_data[symbol].iloc[0]
                position = (initial_capital * equal_weight) / initial_price
                b_position_data[symbol] = position
        
        # Calculate portfolio values
        b_value_data = price_data * b_position_data
        b_value_data['TOTAL'] = b_value_data.drop('TOTAL', axis=1).sum(axis=1)
        
        # Calculate weights
        b_weight_data = b_value_data.div(b_value_data['TOTAL'], axis=0)
        b_weight_data['TOTAL'] = 1.0
        
        # Calculate log returns
        b_log_returns_data = np.log(price_data / price_data.shift(1))
        b_log_returns_data['TOTAL'] = np.log(b_value_data['TOTAL'] / b_value_data['TOTAL'].shift(1))
        
        # Create SMAs for demonstration
        sma_1m = price_data.rolling(window=30).mean()
        sma_3m = price_data.rolling(window=90).mean()
        
        # Build the multi-index structure
        frames = []
        metrics = ['Price', 'B_Position', 'B_Value', 'B_Weight', 'B_log_returns', 'SMA_1m', 'SMA_3m']
        data_frames = [price_data, b_position_data, b_value_data, b_weight_data, b_log_returns_data, sma_1m, sma_3m]
        
        for metric, df in zip(metrics, data_frames):
            # Stack the DataFrame and add metric level
            stacked = df.T.assign(Metric=metric).set_index('Metric', append=True)
            stacked.index = stacked.index.reorder_levels([1, 0])  # Metric, Symbol
            frames.append(stacked)
        
        # Combine all metrics
        combined = pd.concat(frames)
        main_dataframes = combined.T  # Transpose back to have dates as rows
        
        # Set up the column selectors
        first_level_columns = [*set(main_dataframes.columns.get_level_values(0))]
        second_level_columns = sorted([*set(main_dataframes.columns.get_level_values(1))])
        
        st.success("Demo mode enabled! You can now explore the functionality with mock data.")
    else:
        st.stop()
else:
    # We have valid symbols, so demo mode is not needed
    use_demo_mode = False

# Continue with the page logic only if we have data (either real or demo)
if valid_symbols or (not valid_symbols and use_demo_mode):
    
    st.multiselect(
        "Select columns",
        options=first_level_columns,
        default=first_level_columns,
        key="first_level_columns"
    )

    st.multiselect(
        "Select columns",
        options=second_level_columns,
        default=['TOTAL', 'AAPL', 'SPY'] if 'TOTAL' in second_level_columns and 'AAPL' in second_level_columns and 'SPY' in second_level_columns else second_level_columns[:3],
        key="second_level_columns"
    )

    selected_columns = []
    for first_level in st.session_state.first_level_columns:
        for second_level in st.session_state.second_level_columns:
            if (first_level, second_level) in main_dataframes.columns:
                selected_columns.append((first_level, second_level))


    filtered_main_dataframes = main_dataframes.loc[:, selected_columns]

    st.dataframe(filtered_main_dataframes, use_container_width=True)

    # Show transposed tail for easier reading
    st.subheader("Latest Values (Transposed)")
    st.dataframe(filtered_main_dataframes.tail().T, use_container_width=True)

    st.subheader("Complete Dataset")
    st.dataframe(main_dataframes, use_container_width=True)

    # Add some useful analysis from the commented code
    st.subheader("Data Analysis")

    # Get basic information about the data
    first_date = tickers_history.index[0]
    last_date = tickers_history.index[-1]
    days_duration = (last_date - first_date).days + 1
    years_duration = days_duration / 365.25
    total_points = tickers_history.shape[0]
    points_per_year = total_points / years_duration

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("First Date", first_date.strftime('%Y-%m-%d'))
    with col2:
        st.metric("Last Date", last_date.strftime('%Y-%m-%d'))
    with col3:
        st.metric("Years Duration", f"{years_duration:.2f}")
    with col4:
        st.metric("Points per Year", f"{points_per_year:.1f}")

    # Create price charts for individual symbols
    st.subheader("Individual Symbol Analysis")

    # Get available symbols for price analysis
    price_columns = [col[1] for col in main_dataframes.columns if col[0] == 'Price' and col[1] != 'TOTAL']

    selected_symbol = st.selectbox(
        "Select a Symbol for Analysis",
        options=price_columns,
        index=0 if price_columns else None
    )

    if selected_symbol:
        # Get price and fitted data for the selected symbol
        price_data = main_dataframes[('Price', selected_symbol)]
        
        # Check if we have SMA data
        sma_columns = [col for col in main_dataframes.columns if col[0].startswith('SMA_') and col[1] == selected_symbol]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{selected_symbol} Price Analysis**")
            
            # Create a simple line chart for price
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data.values,
                mode='lines',
                name=f'{selected_symbol} Price',
                line=dict(color='blue')
            ))
            
            # Add SMA lines if available
            colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
            for i, sma_col in enumerate(sma_columns[:6]):  # Limit to 6 SMAs
                sma_data = main_dataframes[sma_col]
                fig.add_trace(go.Scatter(
                    x=sma_data.index,
                    y=sma_data.values,
                    mode='lines',
                    name=sma_col[0],
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))
            
            fig.update_layout(
                title=f'{selected_symbol} Price with Moving Averages',
                xaxis_title='Date',
                yaxis_title='Price',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write(f"**{selected_symbol} Statistics**")
            
            # Calculate some basic statistics
            current_price = price_data.iloc[-1]
            first_price = price_data.iloc[0]
            price_change = (current_price / first_price) ** (1/years_duration) - 1
            
            st.metric("Current Price", f"${current_price:.2f}")
            st.metric("First Price", f"${first_price:.2f}")
            st.metric("Annualized Return", f"{price_change:.2%}")
            
            # Show recent price movements
            st.write("**Recent Prices (Last 10 days)**")
            recent_prices = price_data.tail(10)
            st.dataframe(recent_prices.to_frame(), use_container_width=True)

    # Portfolio Performance Analysis
    st.subheader("Portfolio Performance Analysis")

    # Get portfolio value data (TOTAL)
    portfolio_value = main_dataframes[('B_Value', 'TOTAL')]
    portfolio_returns = main_dataframes[('B_log_returns', 'TOTAL')]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Portfolio Value Over Time**")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Total Value',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Portfolio Statistics**")
        
        # Calculate portfolio metrics
        initial_value = portfolio_value.iloc[0]
        current_value = portfolio_value.iloc[-1]
        total_return = (current_value / initial_value) - 1
        annualized_return = (current_value / initial_value) ** (1/years_duration) - 1
        
        # Calculate volatility from log returns
        portfolio_returns_clean = portfolio_returns.dropna()
        if len(portfolio_returns_clean) > 0:
            daily_vol = portfolio_returns_clean.std()
            annualized_vol = daily_vol * (points_per_year ** 0.5)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        else:
            annualized_vol = 0
            sharpe_ratio = 0
        
        st.metric("Initial Value", f"${initial_value:.2f}")
        st.metric("Current Value", f"${current_value:.2f}")
        st.metric("Total Return", f"{total_return:.2%}")
        st.metric("Annualized Return", f"{annualized_return:.2%}")
        st.metric("Annualized Volatility", f"{annualized_vol:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # Individual Asset Performance Comparison
    st.subheader("Asset Performance Comparison")

    # Get all asset values and calculate returns
    asset_symbols = [col[1] for col in main_dataframes.columns if col[0] == 'B_Value' and col[1] != 'TOTAL']

    if asset_symbols:
        performance_data = []
        
        for symbol in asset_symbols:
            asset_value = main_dataframes[('B_Value', symbol)]
            if len(asset_value) > 0:
                initial_val = asset_value.iloc[0]
                current_val = asset_value.iloc[-1]
                if initial_val > 0:
                    total_ret = (current_val / initial_val) - 1
                    annual_ret = (current_val / initial_val) ** (1/years_duration) - 1
                    performance_data.append({
                        'Symbol': symbol,
                        'Initial Value': initial_val,
                        'Current Value': current_val,
                        'Total Return': total_ret,
                        'Annualized Return': annual_ret
                    })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Format the DataFrame for display
            perf_df_display = perf_df.copy()
            perf_df_display['Initial Value'] = perf_df_display['Initial Value'].apply(lambda x: f"${x:.2f}")
            perf_df_display['Current Value'] = perf_df_display['Current Value'].apply(lambda x: f"${x:.2f}")
            perf_df_display['Total Return'] = perf_df_display['Total Return'].apply(lambda x: f"{x:.2%}")
            perf_df_display['Annualized Return'] = perf_df_display['Annualized Return'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(perf_df_display, use_container_width=True, hide_index=True)
            
            # Create a bar chart of annualized returns
            fig = go.Figure(data=[
                go.Bar(x=perf_df['Symbol'], y=perf_df['Annualized Return'], 
                       name='Annualized Return',
                       marker_color='lightblue')
            ])
            
            fig.update_layout(
                title='Annualized Returns by Asset',
                xaxis_title='Symbol',
                yaxis_title='Annualized Return',
                height=400,
                yaxis=dict(tickformat='%')
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No data available to display. Please check your internet connection or try again later.")


# CURRENCY_SYMBOLS = [
#     "EURUSD=X",  # Canadian Dollar to US Dollar
# ]

# # Fetch the tickers data
# tickers_data = tickers_yf(STOCK_SYMBOLS)
# if isinstance(tickers_data, str):
#     st.error(tickers_data)
#     st.stop()

# # Get the tickers history
# tickers_history = tickers_data["history"]
# # Remove null columns
# tickers_history = tickers_history.dropna(axis=1, how='all')

# # Remove null rows
# tickers_history = tickers_history.dropna(axis=0, how='any')


# # Apply rolling exponential fit to each column
# # tickers_history_fitted = tickers_history.apply(get_rolling_exp_fit)
# tickers_history_fitted = tickers_history.rolling(50).mean()

# # Combine the dataframes
# tickers_history = pd.concat(
#     [tickers_history, tickers_history_fitted],
#     axis=1,
#     keys=["Close", "Fitted_Max"],
#     names=["Metric", "Symbol"]
# )


# # add log returns
# log_returns = tickers_history.Close.apply(
#     lambda x: np.log(x / x.shift(1))).dropna()
# log_returns.columns = pd.MultiIndex.from_product(
#     [["Log_Returns"], log_returns.columns], names=["Metric", "Symbol"])

# tickers_history = pd.concat(
#     [tickers_history, log_returns], axis=1, )


# # print(tickers_history.columns)


# # do over/under each day
# over_under = tickers_history["Close"] / tickers_history["Fitted_Max"] - 1
# over_under.columns = pd.MultiIndex.from_product(
#     [["Over/Under_Max"], over_under.columns])
# tickers_history = pd.concat(
#     [tickers_history, over_under], axis=1, names=["Metric", "Symbol"])


# # over/under delta
# tickers_history_fitted_delta = tickers_history_fitted.diff()
# tickers_history_fitted_delta.columns = pd.MultiIndex.from_product(
#     [["Fitted_Delta"], tickers_history_fitted_delta.columns])
# # keep Fitted_Delta within q1 and q3
# q1 = tickers_history_fitted_delta.quantile(0.25, axis=0)
# q3 = tickers_history_fitted_delta.quantile(0.75, axis=0)
# tickers_history_fitted_delta = tickers_history_fitted_delta.clip(
#     lower=q1, upper=q3, axis=1)


# tickers_history = pd.concat(
#     [tickers_history, tickers_history_fitted_delta], axis=1)


# # print the dataframe
# st.dataframe(tickers_history.head(), use_container_width=True)
# st.dataframe(tickers_history.tail(), use_container_width=True)

# # print only the XLC columns
# xlc_df = tickers_history.loc[:,
#                              tickers_history.columns.get_level_values(1) == "XLC"]

# st.dataframe(xlc_df.head(), use_container_width=True)
# st.dataframe(xlc_df.tail(), use_container_width=True)

# xlc_df.columns = xlc_df.columns.droplevel(1)
# fig = display_trend_go_chart_2(
#     xlc_df, "Close", "Fitted_Max")
# st.plotly_chart(fig, use_container_width=True)


# # get points, years, and years duration
# first_date = tickers_history.index[0]
# last_date = tickers_history.index[-1]
# days_duration = (last_date - first_date).days + 1
# years_duration = days_duration / 365.25
# total_points = tickers_history.shape[0]
# points_per_year = total_points / years_duration


# # print the combined dataframe
# # print(tickers_history_combined.head())


# # symbol_data = fetch_symbol_data(STOCK_SYMBOLS, "10y")


# # def extract_invalid_symbols(p_symbol_data: dict):
# #     """
# #     Extract invalid symbols from the symbol data
# #     """
# #     return [data["symbol"] for data in p_symbol_data["infos"] if data["info"] is None or "symbol" not in data["info"]]


# # invalid_symbols = extract_invalid_symbols(symbol_data)


# # def remove_invalid_symbols(p_symbol_data: dict, p_invalid_symbols: list[str]):
# #     """
# #     Remove invalid symbols from the symbol data
# #     """
# #     return {
# #         "history": p_symbol_data["history"].loc[
# #             :,
# #             ~p_symbol_data["history"].columns.get_level_values(1).isin(p_invalid_symbols)],
# #         "infos": [data for data in p_symbol_data["infos"] if data["symbol"] not in p_invalid_symbols]
# #     }


# # valid_symbol_data = remove_invalid_symbols(symbol_data, invalid_symbols)


# # history_close: pd.DataFrame = valid_symbol_data["history"]["Close"].copy()
# # tickers_info = valid_symbol_data["infos"]


# # infos_df = pd.DataFrame(
# #     [(ticker_info['symbol'], ticker_info['info'].get('longName', 'No short name available'))
# #         for ticker_info in tickers_info],
# #     columns=['Symbol', 'Short Name'])
# # st.dataframe(infos_df, use_container_width=True, hide_index=True)


# # history_close.dropna(axis=0, how='any', inplace=True)
# # # print('History Close Data:')
# # # print(history_close)

# # first_date = history_close.index[0]
# # last_date = history_close.index[-1]
# # days_duration = (last_date - first_date).days + 1
# # years_duration = days_duration / 365.25
# # total_points = history_close.shape[0]
# # points_per_year = total_points / years_duration
# # print(f"First date: {first_date}")
# # print(f"Last date: {last_date}")
# # print(f"Days duration: {days_duration}")
# # print(f"Years duration: {years_duration}")
# # print(f"Total points: {total_points}")
# # print(f"Points per year: {points_per_year}")


# # fig, ax = plt.subplots()
# # history_close.plot(ax=ax, title="US Market Historical Close Prices")
# # ax.set_ylabel("Price (USD)")
# # ax.set_xlabel("Date")
# # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
# # plt.tight_layout()
# # st.pyplot(fig)

# # history_close_normalized = history_close.copy()
# # history_close_normalized = history_close_normalized.div(
# #     history_close_normalized.iloc[0, :], axis=1).mul(100)

# # fig, ax = plt.subplots()
# # history_close_normalized.plot(ax=ax, title="US Market Normalized Close Prices")
# # ax.set_ylabel("Normalized Price")
# # ax.set_xlabel("Date")
# # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
# # plt.tight_layout()
# # st.pyplot(fig)


# # history_close_pct = history_close.pct_change().dropna()


# # mean_returns = history_close_pct.describe().T


# # mean_returns['annualized_mean'] = mean_returns['mean'] * points_per_year
# # mean_returns['annualized_std'] = mean_returns['std'] * (
# #     points_per_year ** 0.5)
# # mean_returns['annualized_sharpe'] = mean_returns['annualized_mean'] / \
# #     mean_returns['annualized_std']


# # # scatter plot of annualized mean vs annualized std
# # fig, ax = plt.subplots()
# # mean_returns.plot.scatter(
# #     x='annualized_std', y='annualized_mean', ax=ax, title="Annualized Mean vs Annualized Std")
# # ax.set_xlabel("Annualized Std")
# # ax.set_ylabel("Annualized Mean")

# # for i, row in mean_returns.iterrows():
# #     ax.annotate(row.name,
# #                 (row['annualized_std'] + 0.002, row['annualized_mean']+0.002))

# # st.pyplot(fig)


# # # Covariance and correlation matrices
# # # cov_matrix = history_close_pct.cov()

# # # corr_matrix = history_close_pct.corr()

# # # # Heatmap of the correlation matrix
# # # fig, ax = plt.subplots()
# # # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# # # ax.set_title("Correlation Matrix of US Market Stocks")
# # # st.pyplot(fig)


# # # Logarithmic returns
# # log_returns = history_close.apply(
# #     lambda x: np.log(x / x.shift(1))).dropna()

# # log_returns_description = log_returns.describe().T


# # log_returns_description['annualized_mean'] = log_returns_description['mean'] * points_per_year
# # log_returns_description['annualized_std'] = log_returns_description['std'] * \
# #     (points_per_year ** 0.5)
# # log_returns_description['annualized_sharpe'] = log_returns_description['annualized_mean'] / \
# #     log_returns_description['annualized_std']


# # # log_returns_description['first'] = history_close.iloc[0, :]
# # # log_returns_description['last'] = history_close.iloc[-1, :]
# # # log_returns_description['validation'] = log_returns_description['first'] * \
# # #     np.exp(log_returns_description['mean'] * (total_points-1))


# # print('Log Returns Description:')
# # print(log_returns_description)

# # # cagr_returns = log_returns_description.copy()
# # # cagr_returns['cagr'] = (history_close.iloc[-1, :] /
# # #                         history_close.iloc[0, :]) ** (1 / years_duration) - 1
# # # cagr_returns = cagr_returns[['cagr']].copy()
# # # print('CAGR Returns:')
# # # print(cagr_returns)


# # history_close_fitted = history_close.apply(get_exp_fitted_data)
# # # print('Fitted Data:')
# # # print(history_close_fitted)

# # squared_errors = ((history_close / history_close_fitted)-1) ** 2
# # # print('Squared Errors:')
# # # print(squared_errors)

# # cagr_metrics = np.sqrt(squared_errors.mean())
# # cagr_metrics = pd.DataFrame(cagr_metrics, columns=['RMSE'])
# # cagr_metrics['CAGR'] = (history_close.iloc[-1, :] /
# #                         history_close.iloc[0, :]) ** (1 / years_duration) - 1
# # cagr_metrics['CAGR_Fitted'] = (history_close_fitted.iloc[-1, :] /
# #                                history_close_fitted.iloc[0, :]) ** (1 / years_duration) - 1

# # cagr_metrics['CAGR-to-RMSE Ratio'] = cagr_metrics['CAGR'] / \
# #     cagr_metrics['RMSE']
# # cagr_metrics['Over/Under Today'] = history_close.iloc[-1, :] / \
# #     history_close_fitted.iloc[-1, :] - 1

# # cagr_metrics['Log Returns (annualised)'] = log_returns_description['annualized_mean']
# # cagr_metrics['Log Returns Std (annualised)'] = log_returns_description['annualized_std']
# # cagr_metrics['Log Returns Sharpe (annualised)'] = log_returns_description['annualized_sharpe']

# # cagr_metrics['First Date'] = history_close.index[0]
# # cagr_metrics['Last Date'] = history_close.index[-1]
# # cagr_metrics['Years Duration'] = years_duration

# # print('CAGR Metrics:')
# # print(cagr_metrics)
# # st.dataframe(cagr_metrics.style.format({
# #     "CAGR": "{:.1%}",
# #     "CAGR_Fitted": "{:.1%}",
# #     "RMSE": "{:.1%}",
# #     "Over/Under Today": "{:.1%}",
# #     "Log Returns (annualised)": "{:.1%}",
# #     "Log Returns Std (annualised)": "{:.1%}",
# #     "Log Returns Sharpe (annualised)": "{:.2f}",
# #     "Years Duration": "{:.2f}",
# #     "First Date": lambda x: x.strftime('%Y-%m-%d'),
# #     "Last Date": lambda x: x.strftime('%Y-%m-%d')}),
# #     use_container_width=True
# # )


# # # Create a multi-index DataFrame combining all metrics
# # metrics_dict = {
# #     'Closing Price': history_close,
# #     'Fitted Price': history_close_fitted,
# #     'Squared_Errors': squared_errors,
# #     'Normalized Price': history_close_normalized,
# #     'Pct_Change': history_close_pct,
# #     'Log_Returns': log_returns,
# # }

# # # Combine all metrics into a multi-index DataFrame
# # combined_metrics = pd.concat(metrics_dict, axis=1).dropna()
# # combined_metrics.columns.names = ['Metric', 'Symbol']


# # # Add symbol selector and interactive chart
# # st.subheader("Interactive Symbol Analysis")

# # # Create a mapping of symbol to display name
# # symbol_display_map = {
# #     row['Symbol']: f"{row['Symbol']} - {row['Short Name']}"
# #     for _, row in infos_df.iterrows()
# # }

# # # Create symbol selector with formatted display names
# # selected_symbol = st.selectbox(
# #     "Select a Symbol",
# #     options=list(symbol_display_map.keys()),
# #     format_func=lambda x: symbol_display_map[x],
# #     index=0  # Default to first symbol
# # )

# # # Create the chart using the new utility function
# # fig = display_symbol_metrics_chart(
# #     combined_metrics,
# #     selected_symbol,
# #     metrics=['Closing Price', 'Fitted Price'],
# #     title="Price Analysis"
# # )
# # st.plotly_chart(fig, use_container_width=True)

# # monthly_close = history_close.resample('ME').last()
