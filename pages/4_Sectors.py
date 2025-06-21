"""Streamlit app to display sector and industry information"""

# pylint: disable=C0103
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import minimize

from utilities.app_yfinance import YF_SECTOR_KEYS, market_yf, yf_ticket_info, tickers_yf
from utilities.utilities import fully_analyze_symbol, metrics, retrieve_sector_data
from utilities.go_charts import display_trend_go_chart_2, display_efficient_frontier_chart

print(f"\n--- Sectors view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="US Market Overview", layout="centered")

DEBUG = True
DYNAMIC_MARKET = True

# User contants
home_currency = "EUR"
data_years = 10

# Dynamic way of taking market symbol
###
###
country = "US"

# Fetch market data
market = market_yf(country)
if market is None and DYNAMIC_MARKET:
    st.error("Error retrieving US market data.")
    st.stop()


# Get first market symbol
market_symbol = market['summary'][list(market['summary'].keys())[
    0]].get('symbol', 'UNDEFINED') if DYNAMIC_MARKET else "^GSPC"

# print market keys
if DEBUG:
    print(f"Market keys: {list(market.keys())}")
    print(f"Market summary keys: {list(market['summary'].keys())}")
    # print(f"First market summary: {market['summary'][list(market['summary'].keys())[
    #     0]]}")
    print(f"Market symbol: {market_symbol}")


market_info = yf_ticket_info(market_symbol)
trade_df, trade_metrics, home_df, home_metrics = fully_analyze_symbol(
    market_symbol, home_currency, data_years
)


# Fetch sector data
sector_data, sector_data_df = retrieve_sector_data(home_currency, data_years)


# UI
st.title("US Market Overview")

st.write(
    f"Home currency: **{home_currency}** || Data window: **{data_years} years**")

#
# MARKET
#
st.write("### Market as a whole")


st.write(
    f"Name: **{market_info['shortName']}** || " +
    f"Trade Currency: **{market_info['currency']}** || " +
    f"Symbol: **{market_symbol}**")

# Market Chart
col1, col2, _ = st.columns([1, 1, 2])
with col1:
    years_to_show = st.selectbox(
        "Years displayed",
        options=[1, 2, 3, 5, 7, 10],
    )

currency_options = ['Trade (USD)', 'Home (' + home_currency + ')']
selected_currency = currency_options[1]

with col2:
    selected_currency = st.selectbox(
        "Currency",
        options=currency_options,
        index=1
    )

filtered_first_date = trade_metrics['last_date'] - pd.DateOffset(years=years_to_show) \
    if selected_currency == currency_options[0] \
    else home_metrics['last_date'] - pd.DateOffset(years=years_to_show)

value_fig = display_trend_go_chart_2(
    trade_df[trade_df.index >= filtered_first_date] if selected_currency == currency_options[0] else home_df[home_df.index >= filtered_first_date],
    'Trade Value' if selected_currency == currency_options[0] else 'Home Value',
    'Trade Value Fitted' if selected_currency == currency_options[0] else 'Home Value Fitted',
    title_name=f"{market_info['shortName']}")

st.plotly_chart(
    value_fig,
    config={
        'staticPlot': True},
    use_container_width=True)

metrics_df = pd.DataFrame({
    "Trade Currency": trade_metrics,
    "Home Currency": home_metrics
})
metrics_df.drop([metrics['start_date'].key,
                 metrics['actual_years_duration'].key,
                metrics['last_date'].key], inplace=True)

metrics_df.index = metrics_df.index.map(lambda x: metrics[x].label)

st.dataframe(metrics_df.style.format(
    formatter=lambda x: f"{x:.1%}" if isinstance(x, float) else x
))


st.write("### Sectors")
st.dataframe(sector_data_df.drop(columns=["Reference Date", "Sample Years"]).style.format(
    {
        "Market Weight": "{:.1%}",
        "Trade Over/Under": "{:.1%}",
        "Trade CAGR": "{:.1%}",
        "Trade CAGR Fitted": "{:.1%}",
        "Trade Annualized Return": "{:.1%}",
        "Trade Annualized Risk": "{:.1%}",
        "Trade Return/Risk Ratio": "{:.2f}",
    }
), hide_index=True, height=sector_data_df.shape[0]*39)


for faulty_sector in [
        sector for sector in sector_data if isinstance(sector, str)]:
    st.warning(faulty_sector)


for sector in sector_data:
    sector_name = sector.get('name', 'UNDEFINED')
    overview = sector.get('overview')
    top_companies = sector.get('top_companies')
    top_etfs = sector.get('top_etfs')

    with st.expander(sector_name, expanded=False):

        st.write(
            f"#### {sector_name}")

        if overview is None:
            st.warning(f"No sector overview found. '{sector_name}'")
        else:
            st.write(
                f'**Market Weight**: {100*overview.get('market_weight', -9.99):.1f}%')

            st.write(
                f'**Description**: {overview.get("description", "No description available.")}')

        if top_companies is None:
            st.warning(f"No top companies found. '{sector_name}'")
        else:
            st.write("**Top Companies**")
            st.dataframe(top_companies)

        if top_etfs is None:
            st.warning(f"No top ETFs found. '{sector_name}'")
        else:
            top_etfs_df = pd.DataFrame.from_dict(
                top_etfs, orient='index', columns=['ETF']).reset_index(names='Symbol')
            st.write("**Top ETFs**")
            st.dataframe(top_etfs_df, hide_index=True)

st.divider()

st.subheader("Efficient Frontier")
risk_free_rate = 0.02
iterations = 10000

# Common data preparation for both methods
symbols = [(list(sector_data[sector_id]['top_etfs'].keys()))[0] for sector_id in range(len(sector_data))] \
    if DYNAMIC_MARKET else ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
if DEBUG:
    print(f"Symbols: {symbols}")

# get list of market weights from sector data
sector_market_weights = [sector_data[sector_id]['overview']['market_weight']
                         for sector_id in range(len(sector_data))]

tickers_data = tickers_yf(symbols, period='10y')

# Remove empty rows (any) and empty columns (all)
close_prices = tickers_data["history"].copy().dropna(
    axis=1, how='all').dropna(axis=0, how='any')

number_of_points = close_prices.shape[0]
number_of_days = (close_prices.index[-1] - close_prices.index[0]).days + 1
number_of_years = number_of_days / 365.25
points_per_year = number_of_points / number_of_years

if DEBUG:
    print(f"tickers_data['history'].shape: {tickers_data['history'].shape}")
    print(f"close_prices.shape: {close_prices.shape}")

log_returns = close_prices.apply(lambda x: np.log(x / x.shift(1))).dropna()

# Calculate expected returns and covariance matrix
expected_returns: pd.Series = log_returns.mean() * round(points_per_year)
cov_matrix: pd.DataFrame = log_returns.cov() * round(points_per_year)

# Calculate benchmark (market) volatility and return using same data period
# Get benchmark data for the same time period as sector data
benchmark_data = tickers_yf([market_symbol], period='10y')
benchmark_close_prices = benchmark_data["history"].copy().dropna(
    axis=1, how='all').dropna(axis=0, how='any')

# Use same time period as sector data
benchmark_close_prices: pd.DataFrame = benchmark_close_prices[(benchmark_close_prices.index >= close_prices.index[0]) &
                                                              (benchmark_close_prices.index <= close_prices.index[-1])]

benchmark_points_per_year = benchmark_close_prices.shape[0] / \
    (((benchmark_close_prices.index[-1] -
     benchmark_close_prices.index[0]).days + 1) / 365.25)

# Calculate benchmark log returns
benchmark_log_returns: pd.DataFrame = benchmark_close_prices.apply(
    lambda x: np.log(x / x.shift(1))).dropna()

# Calculate benchmark volatility and return using same annualization
benchmark_volatility = benchmark_log_returns.std(
).iloc[0] * np.sqrt(round(benchmark_points_per_year))

benchmark_return = benchmark_log_returns.mean(
).iloc[0] * round(benchmark_points_per_year)

# Calculate benchmark Sharpe ratio
benchmark_sharpe = (benchmark_return - risk_free_rate) / \
    benchmark_volatility

with st.expander("Manual creation", expanded=False):

    # Manual approach: Generate many random portfolios and find efficient frontier
    portfolio_returns = []
    portfolio_volatilities = []
    weights_list = []
    portfolio_sharpes = []

    # Generate many random portfolios
    for x in range(iterations):  # More iterations for better frontier
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)

        ret = np.sum(weights * expected_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

        portfolio_returns.append(ret)
        portfolio_volatilities.append(vol)
        portfolio_sharpes.append(sharpe)
        weights_list.append(weights)

        if DEBUG and x < 1:
            print(f"\nIteration: {x}")
            print(f"weights: {weights}")
            print(f"portfolio_returns: {ret}")
            print(f"portfolio_volatilities: {vol}")

    # Find efficient portfolios (Pareto optimal)
    efficient_indices = []
    for i, _ in enumerate(portfolio_returns):
        is_efficient = True
        for j, _ in enumerate(portfolio_returns):
            # Check if portfolio j dominates portfolio i
            # (lower volatility AND higher return)
            if (portfolio_returns[j] > portfolio_returns[i] and
                    portfolio_volatilities[j] <= portfolio_volatilities[i]):
                is_efficient = False
                break
        if is_efficient:
            efficient_indices.append(i)

    # Sort efficient portfolios by volatility to get the frontier curve
    efficient_indices.sort(key=lambda i: portfolio_volatilities[i])

    # Create efficient frontier DataFrame

    efficient_frontier_df = pd.DataFrame({
        "Returns": [portfolio_returns[i] for i in efficient_indices],
        "Volatility": [portfolio_volatilities[i] for i in efficient_indices]
    })

    # Create random portfolios DataFrame for comparison
    random_df = pd.DataFrame({
        "Returns": portfolio_returns,
        "Volatility": portfolio_volatilities
    })

    # Find portfolios with higher Sharpe ratio than benchmark
    better_sharpe_indices = [i for i, sharpe in enumerate(
        portfolio_sharpes) if sharpe > benchmark_sharpe]

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = max(range(len(portfolio_sharpes)),
                         key=lambda i: portfolio_sharpes[i])
    max_sharpe_portfolio = {
        'return': portfolio_returns[max_sharpe_idx],
        'volatility': portfolio_volatilities[max_sharpe_idx],
        'sharpe': portfolio_sharpes[max_sharpe_idx],
        'weights': weights_list[max_sharpe_idx]
    }

    # Find portfolio on efficient frontier closest to benchmark volatility
    # Look for portfolios with volatility lower than benchmark
    lower_vol_portfolios = efficient_frontier_df[efficient_frontier_df['Volatility']
                                                 < benchmark_volatility]
    # Look for portfolios with volatility higher than benchmark
    higher_vol_portfolios = efficient_frontier_df[efficient_frontier_df['Volatility']
                                                  > benchmark_volatility]

    if not lower_vol_portfolios.empty and not higher_vol_portfolios.empty:
        # Take the highest volatility portfolio below benchmark
        lower_vol_portfolio = lower_vol_portfolios.iloc[-1]
        # Take the lowest volatility portfolio above benchmark
        higher_vol_portfolio = higher_vol_portfolios.iloc[0]
        # Use the middle value
        target_volatility = (
            lower_vol_portfolio['Volatility'] + higher_vol_portfolio['Volatility']) / 2
    elif not lower_vol_portfolios.empty:
        # Only lower volatility portfolios available
        target_volatility = lower_vol_portfolios.iloc[-1]['Volatility']
    elif not higher_vol_portfolios.empty:
        # Only higher volatility portfolios available
        target_volatility = higher_vol_portfolios.iloc[0]['Volatility']
    else:
        # Fallback to closest volatility
        target_volatility = benchmark_volatility

    # Find the portfolio on efficient frontier closest to target volatility
    closest_idx = efficient_frontier_df['Volatility'].sub(
        target_volatility).abs().idxmin()
    same_risk_portfolio = {
        'return': efficient_frontier_df.loc[closest_idx, 'Returns'],
        'volatility': efficient_frontier_df.loc[closest_idx, 'Volatility'],
        'weights': weights_list[efficient_indices[closest_idx]],
        'vol_diff': abs(efficient_frontier_df.loc[closest_idx, 'Volatility'] - benchmark_volatility)
    }

    # Calculate Sharpe ratio for same risk portfolio
    same_risk_portfolio['sharpe'] = (same_risk_portfolio['return'] - risk_free_rate) / \
        same_risk_portfolio['volatility'] if same_risk_portfolio['volatility'] > 0 else 0

    # Find portfolio on efficient frontier with similar risk but higher return than benchmark
    # First, find portfolios with similar volatility (within 1% tolerance)
    volatility_tolerance = 0.01
    similar_risk_portfolios = efficient_frontier_df[
        (efficient_frontier_df['Volatility'] >= benchmark_volatility - volatility_tolerance) &
        (efficient_frontier_df['Volatility'] <= benchmark_volatility + volatility_tolerance) &
        (efficient_frontier_df['Returns'] > benchmark_return)
    ]

    if not similar_risk_portfolios.empty:
        # Take the one with highest return among similar risk portfolios
        best_similar_risk = similar_risk_portfolios.loc[similar_risk_portfolios['Returns'].idxmax(
        )]
        same_risk_portfolio = {
            'return': best_similar_risk['Returns'],
            'volatility': best_similar_risk['Volatility'],
            'weights': weights_list[efficient_indices[best_similar_risk.name]],
            'vol_diff': abs(best_similar_risk['Volatility'] - benchmark_volatility)
        }
    else:
        # If no portfolio with similar risk and higher return, find closest volatility with higher return
        higher_return_portfolios = efficient_frontier_df[
            efficient_frontier_df['Returns'] > benchmark_return]
        if not higher_return_portfolios.empty:
            closest_idx = higher_return_portfolios['Volatility'].sub(
                benchmark_volatility).abs().idxmin()
            same_risk_portfolio = {
                'return': higher_return_portfolios.loc[closest_idx, 'Returns'],
                'volatility': higher_return_portfolios.loc[closest_idx, 'Volatility'],
                'weights': weights_list[efficient_indices[closest_idx]],
                'vol_diff': abs(higher_return_portfolios.loc[closest_idx, 'Volatility'] - benchmark_volatility)
            }
        else:
            # Fallback to closest volatility (this shouldn't happen if efficient frontier is properly calculated)
            closest_idx = efficient_frontier_df['Volatility'].sub(
                benchmark_volatility).abs().idxmin()
            same_risk_portfolio = {
                'return': efficient_frontier_df.loc[closest_idx, 'Returns'],
                'volatility': efficient_frontier_df.loc[closest_idx, 'Volatility'],
                'weights': weights_list[efficient_indices[closest_idx]],
                'vol_diff': abs(efficient_frontier_df.loc[closest_idx, 'Volatility'] - benchmark_volatility)
            }

    # Calculate Sharpe ratio for same risk portfolio
    same_risk_portfolio['sharpe'] = (same_risk_portfolio['return'] - risk_free_rate) / \
        same_risk_portfolio['volatility'] if same_risk_portfolio['volatility'] > 0 else 0

    if DEBUG:
        print(f"\n{'='*50}")
        print("PORTFOLIO ANALYSIS")
        print(f"{'='*50}")
        print(f"Risk-free rate: {risk_free_rate:.2%}")
        print(
            f"Data period: {close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"Number of data points: {len(benchmark_log_returns)}")
        print(f"{'='*50}")

        print(f"{'BENCHMARK':<20} {'RETURN':<12} {'VOLATILITY':<12} {'SHARPE':<12}")
        print(f"{'-'*60}")
        print(f"{market_symbol:<20} {benchmark_return:<12.4f} {benchmark_volatility:<12.4f} {benchmark_sharpe:<12.4f}")
        print(f"{'='*50}")

        print(
            f"{'MAX SHARPE PORTFOLIO':<20} {'RETURN':<12} {'VOLATILITY':<12} {'SHARPE':<12}")
        print(f"{'-'*60}")
        print(
            f"{'Optimal Portfolio':<20} {max_sharpe_portfolio['return']:<12.4f} {max_sharpe_portfolio['volatility']:<12.4f} {max_sharpe_portfolio['sharpe']:<12.4f}")
        print(f"{'='*50}")

        print(
            f"Portfolios with higher Sharpe than benchmark: {len(better_sharpe_indices)}")
        print(
            f"Improvement in Sharpe: {max_sharpe_portfolio['sharpe'] - benchmark_sharpe:.4f}")
        print(f"{'='*50}")

        print("WEIGHT COMPARISON (Optimal vs Market):")
        print(f"{'-'*50}")
        for symbol, optimal_weight, market_weight in zip(symbols, max_sharpe_portfolio['weights'], sector_market_weights):
            diff = optimal_weight - market_weight
            print(
                f"{symbol:<10} Optimal: {optimal_weight:>6.1%} | Market: {market_weight:>6.1%} | Diff: {diff:+6.1%}")
        print(f"{'='*50}")

    # plot efficient frontier first
    frontier_fig = display_efficient_frontier_chart(
        efficient_frontier_df,
        random_df=random_df,
        benchmark_return=benchmark_return,
        benchmark_volatility=benchmark_volatility,
        benchmark_name=market_symbol,
        max_sharpe_return=max_sharpe_portfolio['return'],
        max_sharpe_volatility=max_sharpe_portfolio['volatility'],
        max_sharpe_name="Max Sharpe Portfolio",
        same_risk_return=same_risk_portfolio['return'],
        same_risk_volatility=same_risk_portfolio['volatility'],
        same_risk_name="Same Risk Portfolio",
        title_name='Efficient Frontier (Manual)'
    )

    if frontier_fig is not None:
        st.plotly_chart(
            frontier_fig,
            config={'staticPlot': True},
            use_container_width=True
        )
    else:
        st.warning("No valid data to plot for Efficient Frontier.")

    # Display portfolio analysis results in Streamlit
    st.write("### Portfolio Analysis Results")

    # Consolidated portfolio comparison
    st.write("**Portfolio Performance Comparison**")
    portfolio_comparison = {
        "Portfolio": [market_symbol, "Optimal Portfolio", "Same Risk Portfolio"],
        "Return": [f"{benchmark_return:.2%}", f"{max_sharpe_portfolio['return']:.2%}", f"{same_risk_portfolio['return']:.2%}"],
        "Volatility": [f"{benchmark_volatility:.2%}", f"{max_sharpe_portfolio['volatility']:.2%}", f"{same_risk_portfolio['volatility']:.2%}"],
        "Sharpe Ratio": [f"{benchmark_sharpe:.3f}", f"{max_sharpe_portfolio['sharpe']:.3f}", f"{same_risk_portfolio['sharpe']:.3f}"]
    }
    portfolio_df = pd.DataFrame(portfolio_comparison)
    st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

    # Improvement metrics
    st.write("**Performance Improvement**")
    improvement_metrics = {
        "Metric": [
            "Portfolios with higher Sharpe than benchmark",
            "Improvement in Sharpe ratio",
            "Data period"
        ],
        "Value": [
            f"{len(better_sharpe_indices)}",
            f"{max_sharpe_portfolio['sharpe'] - benchmark_sharpe:.4f}",
            f"{close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}"
        ]
    }
    st.dataframe(pd.DataFrame(improvement_metrics),
                 use_container_width=True, hide_index=True)

    # Portfolio weights comparison
    st.write("**Portfolio Weights Comparison**")

    # Create HTML table
    html_table = """
    <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
        <thead>
            <tr style="border-bottom: 2px solid #ddd;">
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Sector</th>
                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Market</th>
                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Optimal</th>
                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Same Risk</th>
            </tr>
        </thead>
        <tbody>
    """

    for symbol, optimal_weight, market_weight, same_risk_weight in zip(symbols, max_sharpe_portfolio['weights'], sector_market_weights, same_risk_portfolio['weights']):
        html_table += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 6px 8px;">{symbol}</td>
                <td style="text-align: right; padding: 6px 8px;">{market_weight:.1%}</td>
                <td style="text-align: right; padding: 6px 8px;">{optimal_weight:.1%}</td>
                <td style="text-align: right; padding: 6px 8px;">{same_risk_weight:.1%}</td>
            </tr>
        """

    html_table += """
        </tbody>
    </table>
    """

    components.html(html_table, height=500)

with st.expander("Scientific Optimization (scipy)", expanded=False):

    def portfolio_volatility(weights, cov_matrix):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(weights, expected_returns):
        """Calculate portfolio return"""
        return np.sum(weights * expected_returns)

    def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        """Negative Sharpe ratio (for minimization)"""
        ret = portfolio_return(weights, expected_returns)
        vol = portfolio_volatility(weights, cov_matrix)
        return -(ret - risk_free_rate) / vol if vol > 0 else 1e6

    def constraints(weights):
        """Sum of weights must equal 1"""
        return np.sum(weights) - 1

    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(len(symbols)))

    # Initial guess: equal weights
    initial_weights = np.array([1/len(symbols)] * len(symbols))

    # Optimize for maximum Sharpe ratio
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': constraints}
    )

    if result.success:
        optimal_weights = result.x
        optimal_return = portfolio_return(optimal_weights, expected_returns)
        optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)
        optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

        # Create efficient frontier using scipy optimization
        efficient_frontier_points = []
        target_returns = np.linspace(
            expected_returns.min(), expected_returns.max(), 50)

        for target_return in target_returns:
            # Minimize volatility for given return target
            def objective(weights):
                return portfolio_volatility(weights, cov_matrix)

            def return_constraint(weights):
                return portfolio_return(weights, expected_returns) - target_return

            try:
                result_min_vol = minimize(
                    objective,
                    initial_weights,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[
                        {'type': 'eq', 'fun': constraints},
                        {'type': 'eq', 'fun': return_constraint}
                    ]
                )

                if result_min_vol.success:
                    vol = portfolio_volatility(result_min_vol.x, cov_matrix)
                    ret = portfolio_return(result_min_vol.x, expected_returns)
                    efficient_frontier_points.append(
                        {'Returns': ret, 'Volatility': vol})
            except:
                continue

        efficient_frontier_df = pd.DataFrame(efficient_frontier_points)

        # Find same-risk portfolio (closest volatility to benchmark)
        if not efficient_frontier_df.empty:
            closest_idx = efficient_frontier_df['Volatility'].sub(
                benchmark_volatility).abs().idxmin()
            same_risk_portfolio = {
                'return': efficient_frontier_df.loc[closest_idx, 'Returns'],
                'volatility': efficient_frontier_df.loc[closest_idx, 'Volatility'],
                'weights': optimal_weights,  # This would need to be calculated properly
                'vol_diff': abs(efficient_frontier_df.loc[closest_idx, 'Volatility'] - benchmark_volatility)
            }
            same_risk_portfolio['sharpe'] = (same_risk_portfolio['return'] - risk_free_rate) / \
                same_risk_portfolio['volatility'] if same_risk_portfolio['volatility'] > 0 else 0
        else:
            same_risk_portfolio = {
                'return': optimal_return,
                'volatility': optimal_volatility,
                'weights': optimal_weights,
                'sharpe': optimal_sharpe,
                'vol_diff': abs(optimal_volatility - benchmark_volatility)
            }

        # Find same-risk portfolio with higher return than benchmark
        if not efficient_frontier_df.empty:
            # Find portfolios with similar volatility (within 1% tolerance) and higher return
            volatility_tolerance = 0.01
            similar_risk_portfolios = efficient_frontier_df[
                (efficient_frontier_df['Volatility'] >= benchmark_volatility - volatility_tolerance) &
                (efficient_frontier_df['Volatility'] <= benchmark_volatility + volatility_tolerance) &
                (efficient_frontier_df['Returns'] > benchmark_return)
            ]

            if not similar_risk_portfolios.empty:
                # Take the one with highest return among similar risk portfolios
                best_similar_risk = similar_risk_portfolios.loc[similar_risk_portfolios['Returns'].idxmax(
                )]
                same_risk_portfolio = {
                    'return': best_similar_risk['Returns'],
                    'volatility': best_similar_risk['Volatility'],
                    'weights': optimal_weights,  # Would need proper calculation
                    'vol_diff': abs(best_similar_risk['Volatility'] - benchmark_volatility)
                }
            else:
                # If no portfolio with similar risk and higher return, find closest volatility with higher return
                higher_return_portfolios = efficient_frontier_df[
                    efficient_frontier_df['Returns'] > benchmark_return]
                if not higher_return_portfolios.empty:
                    closest_idx = higher_return_portfolios['Volatility'].sub(
                        benchmark_volatility).abs().idxmin()
                    same_risk_portfolio = {
                        'return': higher_return_portfolios.loc[closest_idx, 'Returns'],
                        'volatility': higher_return_portfolios.loc[closest_idx, 'Volatility'],
                        'weights': optimal_weights,  # Would need proper calculation
                        'vol_diff': abs(higher_return_portfolios.loc[closest_idx, 'Volatility'] - benchmark_volatility)
                    }
                else:
                    # Fallback to closest volatility
                    closest_idx = efficient_frontier_df['Volatility'].sub(
                        benchmark_volatility).abs().idxmin()
                    same_risk_portfolio = {
                        'return': efficient_frontier_df.loc[closest_idx, 'Returns'],
                        'volatility': efficient_frontier_df.loc[closest_idx, 'Volatility'],
                        'weights': optimal_weights,  # Would need proper calculation
                        'vol_diff': abs(efficient_frontier_df.loc[closest_idx, 'Volatility'] - benchmark_volatility)
                    }

            # Calculate Sharpe ratio for same risk portfolio
            same_risk_portfolio['sharpe'] = (same_risk_portfolio['return'] - risk_free_rate) / \
                same_risk_portfolio['volatility'] if same_risk_portfolio['volatility'] > 0 else 0
        else:
            same_risk_portfolio = {
                'return': optimal_return,
                'volatility': optimal_volatility,
                'weights': optimal_weights,
                'sharpe': optimal_sharpe,
                'vol_diff': abs(optimal_volatility - benchmark_volatility)
            }

        # Create max Sharpe portfolio object
        max_sharpe_portfolio = {
            'return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe': optimal_sharpe,
            'weights': optimal_weights
        }

        # plot efficient frontier first
        frontier_fig = display_efficient_frontier_chart(
            efficient_frontier_df,
            random_df=None,  # No random portfolios for scientific method
            benchmark_return=benchmark_return,
            benchmark_volatility=benchmark_volatility,
            benchmark_name=market_symbol,
            max_sharpe_return=max_sharpe_portfolio['return'],
            max_sharpe_volatility=max_sharpe_portfolio['volatility'],
            max_sharpe_name="Max Sharpe Portfolio",
            same_risk_return=same_risk_portfolio['return'],
            same_risk_volatility=same_risk_portfolio['volatility'],
            same_risk_name="Same Risk Portfolio",
            title_name='Efficient Frontier (Scientific)'
        )

        if frontier_fig is not None:
            st.plotly_chart(
                frontier_fig,
                config={'staticPlot': True},
                use_container_width=True
            )
        else:
            st.warning("No valid data to plot for Efficient Frontier.")

        # Display portfolio analysis results in Streamlit
        st.write("### Portfolio Analysis Results")

        # Consolidated portfolio comparison
        st.write("**Portfolio Performance Comparison**")
        portfolio_comparison = {
            "Portfolio": [market_symbol, "Optimal Portfolio", "Same Risk Portfolio"],
            "Return": [f"{benchmark_return:.2%}", f"{max_sharpe_portfolio['return']:.2%}", f"{same_risk_portfolio['return']:.2%}"],
            "Volatility": [f"{benchmark_volatility:.2%}", f"{max_sharpe_portfolio['volatility']:.2%}", f"{same_risk_portfolio['volatility']:.2%}"],
            "Sharpe Ratio": [f"{benchmark_sharpe:.3f}", f"{max_sharpe_portfolio['sharpe']:.3f}", f"{same_risk_portfolio['sharpe']:.3f}"]
        }
        portfolio_df = pd.DataFrame(portfolio_comparison)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

        # Improvement metrics
        st.write("**Performance Improvement**")
        improvement_metrics = {
            "Metric": [
                "Improvement in Sharpe ratio",
                "Data period"
            ],
            "Value": [
                f"{max_sharpe_portfolio['sharpe'] - benchmark_sharpe:.4f}",
                f"{close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}"
            ]
        }
        st.dataframe(pd.DataFrame(improvement_metrics),
                     use_container_width=True, hide_index=True)

        # Portfolio weights comparison
        st.write("**Portfolio Weights Comparison**")

        # Create HTML table
        html_table = """
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead>
                <tr style="border-bottom: 2px solid #ddd;">
                    <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Sector</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Market</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Optimal</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">Same Risk</th>
                </tr>
            </thead>
            <tbody>
        """

        for symbol, optimal_weight, market_weight, same_risk_weight in zip(symbols, max_sharpe_portfolio['weights'], sector_market_weights, same_risk_portfolio['weights']):
            html_table += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px 8px;">{symbol}</td>
                    <td style="text-align: right; padding: 6px 8px;">{market_weight:.1%}</td>
                    <td style="text-align: right; padding: 6px 8px;">{optimal_weight:.1%}</td>
                    <td style="text-align: right; padding: 6px 8px;">{same_risk_weight:.1%}</td>
                </tr>
            """

        html_table += """
            </tbody>
        </table>
        """

        components.html(html_table, height=500)

    else:
        st.error("Optimization failed!")
        st.write(f"Error: {result.message}")
