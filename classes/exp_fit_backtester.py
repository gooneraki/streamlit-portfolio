"""
Exponential Fit Backtester Module

This module provides a sophisticated backtesting framework for portfolio optimization
using exponential fitting techniques. It implements a mean-reversion strategy based
on rolling exponential fits to identify deviations from trend and adjust portfolio weights.

Classes:
    ExpFitBacktester: Main backtesting class for portfolio optimization
"""

from typing import Literal, Union, Optional, Dict, Any, List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from utilities.app_yfinance import tickers_yf, TickersData


@dataclass
class BacktestResults:
    """Container for backtest performance metrics and data"""
    benchmark_performance: float
    portfolio_performance: float
    outperformance: float
    consolidated_data: pd.DataFrame
    main_data_multiindex: pd.DataFrame
    tickers_history: pd.DataFrame
    tickers_news: Dict[str, List[Dict[str, Any]]]


def print_df(df: pd.DataFrame, title: str) -> None:
    """
    Utility function to print DataFrame information for debugging.

    Args:
        df: DataFrame to print
        title: Title for the output
    """
    print(f"\n{title}")
    print(df.head())
    print(df.tail())
    print(f"Shape: {df.shape}")


def get_central_deviations(close_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate central deviations from exponential fit trends.

    This function applies rolling exponential fits to price data and calculates
    how much each price deviates from its fitted trend, then centers these deviations.

    Args:
        close_data: DataFrame with close prices for multiple tickers

    Returns:
        DataFrame with centered deviations from exponential fits
    """
    # Apply rolling exponential fit with minimum 3 years of data
    # min_points = round(365.25 * 3)
    # tickers_history_fitted = close_data.apply(
    #     get_rolling_exp_fit, min_points=min_points
    # )
    tickers_history_fitted = close_data.rolling(window=50).mean()

    # Calculate deviations from fitted values
    deviations = (close_data/tickers_history_fitted - 1).dropna()

    # Center deviations by subtracting mean across tickers for each date
    central_deviations = deviations.sub(deviations.mean(axis=1), axis=0)

    return central_deviations


def get_portfolio_weights(
    benchmark_weights: pd.DataFrame,
    central_deviations: pd.DataFrame,
    tilt: float,
    period: Literal['weekly', 'monthly']
) -> pd.DataFrame:
    """
    Calculate portfolio weights based on central deviations and tilt factor.

    This function combines benchmark weights with tilted weights based on
    central deviations, then rebalances at specified intervals.

    Args:
        benchmark_weights: Equal-weighted benchmark portfolio weights
        central_deviations: Centered deviations from exponential fits
        tilt: Factor to control the strength of the tilt (0 = no tilt, 1 = full tilt)
        period: Rebalancing frequency ('weekly' or 'monthly')

    Returns:
        DataFrame with optimized portfolio weights
    """
    # Calculate tilted weights by combining benchmark with deviations
    portfolio_weights_pre = (benchmark_weights + tilt *
                             central_deviations).dropna()

    # Resample to specified frequency and take last value of each period
    freq = 'W' if period == 'weekly' else 'ME'
    weights_last_period = portfolio_weights_pre.resample(freq).last()

    # Forward fill weights to daily frequency
    weights_reindexed = weights_last_period.reindex(
        central_deviations.index, method='ffill'
    ).dropna()

    # Create final portfolio weights by combining with benchmark
    portfolio_weights = benchmark_weights.copy()
    portfolio_weights.loc[weights_reindexed.index,
                          weights_reindexed.columns] = weights_reindexed

    return portfolio_weights


class ExpFitBacktester:
    """
    Exponential Fit Backtester for portfolio optimization.

    This class implements a sophisticated backtesting framework that uses
    rolling exponential fits to identify mean-reversion opportunities in
    financial markets. It compares the performance of an optimized portfolio
    against an equal-weighted benchmark.

    Attributes:
        symbols: List of ticker symbols to backtest
        tilt: Factor controlling the strength of portfolio tilts (default: 0.5)
        column_titles: Column names for multi-index DataFrames
    """

    column_titles = ["Metric", "Ticker"]
    tilt = 0.5

    def __init__(self, symbols: List[str], initial_capital: float = 1000):
        """
        Initialize the backtester with a list of symbols and initial capital.

        Args:
            symbols: List of ticker symbols to include in the backtest
            initial_capital: Starting capital for the backtest (default: 1000)

        Raises:
            ValueError: If no valid symbols are found or data fetching fails
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.tickers_data: Optional[Union[str, TickersData]] = None
        self.valid_symbols: List[str] = []
        # MultiIndex DataFrame with (Metric, Symbol) columns
        self.main_data_multiindex: Optional[pd.DataFrame] = None

        self._fetch_data()
        self._preprocess_data()

    def _fetch_data(self) -> None:
        """
        Fetch market data for all symbols.

        Raises:
            ValueError: If data fetching fails
        """
        self.tickers_data = tickers_yf(self.symbols)

        if isinstance(self.tickers_data, str):
            raise ValueError(f"Failed to fetch data: {self.tickers_data}")

    def _preprocess_data(self) -> None:
        """
        Preprocess and validate the fetched data.

        This method cleans the data, calculates all necessary metrics,
        and prepares the backtest results.

        Raises:
            ValueError: If no valid symbols are found or calculations fail
        """
        if not isinstance(self.tickers_data, dict):
            raise ValueError("Invalid tickers data format")

        # Clean the historical data
        history_data = self.tickers_data["history"]
        history_data = history_data.dropna(
            axis=1, how='all')  # Remove empty columns
        history_data = history_data.dropna(
            axis=0, how='any')   # Remove rows with missing data

        # Update tickers data with cleaned history
        self.tickers_data["history"] = history_data

        # Get valid symbols
        self.valid_symbols = history_data.columns.tolist()

        if not self.valid_symbols:
            raise ValueError("No valid symbols found after data cleaning")

        # Calculate all metrics
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        """
        Calculate all performance metrics and create consolidated data structures.
        Only keep a single MultiIndex DataFrame with benchmark metrics.
        """
        if not isinstance(self.tickers_data, dict):
            raise ValueError("Invalid tickers data format")

        close_data = self.tickers_data["history"]

        # The below are SERIES with the symbols as index (e.g. [11 symbols, 1 column])
        initial_weights = pd.Series(
            1 / len(self.valid_symbols), index=close_data.columns)
        initial_prices: pd.Series = close_data.loc[close_data.index[0]]
        initial_benchmark_positions = (self.initial_capital *
                                       initial_weights) / initial_prices

        # The below are DATAFRAMES with the symbols as columns (e.g. [1000 dates, 11 symbols])
        price_df = close_data.copy()
        b_position_df = pd.DataFrame([initial_benchmark_positions] * len(close_data),
                                     index=close_data.index,
                                     columns=close_data.columns)

        b_value_df = price_df * b_position_df
        total_b_value = b_value_df.sum(axis=1)
        b_weight_df = b_value_df.div(total_b_value, axis=0)
        b_log_returns_df = np.log(price_df / price_df.shift(1))

        # Add TOTAL column for benchmark
        b_value_df["TOTAL"] = total_b_value
        b_weight_df["TOTAL"] = b_weight_df.sum(axis=1)
        b_log_returns_df["TOTAL"] = np.log(
            total_b_value / total_b_value.shift(1))

        # Stack into MultiIndex DataFrame
        metrics = ["Price", "B_Position",
                   "B_Value", "B_Weight", "B_log_returns"]

        frames = [price_df, b_position_df,
                  b_value_df, b_weight_df, b_log_returns_df]
        #   mean_reversion_signal, return_spread_signal, return_spread_signal_mean,
        #   volatility_signal, sharpe_signal, relative_strength_signal]
        arrays = []
        for metric, df in zip(metrics, frames):
            arrays.append(pd.DataFrame(df, columns=df.columns).T.assign(
                Metric=metric).set_index("Metric", append=True))
        stacked = pd.concat(arrays)

        # Fix MultiIndex reordering
        if isinstance(stacked.index, pd.MultiIndex):
            stacked.index = stacked.index.reorder_levels(
                [1, 0])  # Metric, Symbol
        else:
            # Alternative approach for older pandas versions
            stacked = stacked.swaplevel(0, 1)

        stacked = stacked.sort_index()

        # Dates as rows, (Metric, Symbol) as columns
        self.main_data_multiindex = stacked.T

        # Validate return calculations for benchmark TOTAL and portfolio TOTAL
        self._validate_return_calculations(
            self.main_data_multiindex[("B_Value", "TOTAL")],
            self.main_data_multiindex[("B_log_returns", "TOTAL")],
            prefix="Benchmark (TOTAL)"
        )

        # In case we want to validate for each symbol
        # for symbol in self.main_data_multiindex["B_Value"].columns:
        #     self._validate_return_calculations(
        #         self.main_data_multiindex[("B_Value", symbol)],
        #         self.main_data_multiindex[("B_log_returns", symbol)],
        #         prefix=f"Benchmark ({symbol})"
        #     )

        self._calculate_signals()

    def _get_points_per_month(self) -> float:
        """
        Calculate points per month based on the data frequency.

        Returns:
            Number of data points per month
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        first_date = self.main_data_multiindex.index[0]
        last_date = self.main_data_multiindex.index[-1]
        days_count = (last_date - first_date).days + 1
        years_count = days_count / 365.25
        total_points = len(self.main_data_multiindex.index)
        points_per_month = total_points / (years_count * 12)

        return points_per_month

    def _calculate_signals(self) -> None:
        """
        Calculate all signals for the main data multiindex.
        """

        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        points_per_month = self._get_points_per_month()

        sma_1m = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month)).mean()
        sma_3m = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month * 3)).mean()
        sma_6m = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month * 6)).mean()
        sma_1y = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month * 12)).mean()
        sma_2y = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month * 24)).mean()

        sma_3y = self.main_data_multiindex["Price"].rolling(
            window=round(points_per_month * 36)).mean()

        # Collect all SMAs and add them at once
        sma_dataframes = []
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_1m, "SMA_1m"))
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_3m, "SMA_3m"))
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_6m, "SMA_6m"))
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_1y, "SMA_1y"))
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_2y, "SMA_2y"))
        sma_dataframes.append(
            self._prepare_sma_for_multiindex(sma_3y, "SMA_3y"))

        # Concatenate all SMAs at once
        if sma_dataframes:
            all_smas = pd.concat(sma_dataframes, axis=1)
            self.main_data_multiindex = pd.concat(
                [self.main_data_multiindex, all_smas], axis=1)

    def _prepare_sma_for_multiindex(self, sma_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """
        Prepare a simple moving average DataFrame for multiindex format.

        Args:
            sma_df: DataFrame with SMA values
            metric_name: Name to be used for the Metric level

        Returns:
            DataFrame ready for multiindex concatenation
        """
        # Convert to multiindex format
        sma_multiindex = sma_df.T.assign(
            Metric=metric_name).set_index("Metric", append=True)

        # Fix MultiIndex reordering
        if isinstance(sma_multiindex.index, pd.MultiIndex):
            sma_multiindex.index = sma_multiindex.index.reorder_levels(
                [1, 0])  # Metric, Symbol
        else:
            # Alternative approach for older pandas versions
            sma_multiindex = sma_multiindex.swaplevel(0, 1)

        return sma_multiindex.T

    def get_mean_reversion_signal(self, price_df: pd.DataFrame, total_b_value: pd.Series, window: int = 42) -> pd.DataFrame:
        """
        Calculate the mean reversion signal (Z-score) for each symbol using a rolling mean and std.
        For TOTAL, use the portfolio value column instead of NaN.
        Args:
            price_df: DataFrame of prices (should include all symbols and TOTAL)
            window: Rolling window size (default 42)
        Returns:
            DataFrame of Z-scores (mean reversion signal) for each symbol, and for TOTAL (portfolio value)
        """
        mean_rolling = price_df[self.valid_symbols].rolling(
            window=window).mean()
        std_rolling = price_df[self.valid_symbols].rolling(window=window).std()
        mean_reversion_signal = (
            price_df[self.valid_symbols] - mean_rolling) / std_rolling

        total_series = total_b_value
        total_mean = total_series.rolling(window=window).mean()
        total_std = total_series.rolling(window=window).std()
        mean_reversion_signal["TOTAL"] = (
            total_series - total_mean) / total_std
        return mean_reversion_signal

    def get_return_spread_signal(self, b_log_returns_df: pd.DataFrame, short_window: int = 21, long_window: int = 126) -> pd.DataFrame:
        """
        Calculate the Z-score of the difference between short-term and long-term rolling sums of B_log_returns.
        Args:
            b_log_returns_df: DataFrame of log returns (should include all symbols and TOTAL)
            short_window: Short rolling window (default 21)
            long_window: Long rolling window (default 126)
        Returns:
            DataFrame of Z-scores (return spread signal) for each symbol and TOTAL
        """
        short_sum = b_log_returns_df[self.valid_symbols].rolling(
            window=short_window).sum()
        long_sum = b_log_returns_df[self.valid_symbols].rolling(
            window=long_window).sum()
        spread = short_sum - long_sum
        spread_mean = spread.rolling(window=long_window).mean()
        spread_std = spread.rolling(window=long_window).std()
        spread_zscore = (spread - spread_mean) / spread_std
        # For TOTAL
        short_sum_total = b_log_returns_df["TOTAL"].rolling(
            window=short_window).sum()
        long_sum_total = b_log_returns_df["TOTAL"].rolling(
            window=long_window).sum()
        spread_total = short_sum_total - long_sum_total
        spread_mean_total = spread_total.rolling(window=long_window).mean()
        spread_std_total = spread_total.rolling(window=long_window).std()
        spread_zscore["TOTAL"] = (
            spread_total - spread_mean_total) / spread_std_total
        return spread_zscore

    def get_return_spread_signal_mean(self, log_returns_df: pd.DataFrame, short_window: int = 21, long_window: int = 126) -> pd.DataFrame:
        """
        Calculate the Z-score of the difference between short-term and long-term rolling MEANS of B_log_returns.
        Args:
            log_returns_df: DataFrame of log returns (should include all symbols and TOTAL)
            short_window: Short rolling window (default 21)
            long_window: Long rolling window (default 126)
        Returns:
            DataFrame of Z-scores (return spread signal using means) for each symbol and TOTAL
        """
        short_mean = log_returns_df[self.valid_symbols].rolling(
            window=short_window).mean()
        long_mean = log_returns_df[self.valid_symbols].rolling(
            window=long_window).mean()
        spread = short_mean - long_mean
        spread_mean = spread.rolling(window=long_window).mean()
        spread_std = spread.rolling(window=long_window).std()
        spread_zscore = (spread - spread_mean) / spread_std
        # For TOTAL
        short_mean_total = log_returns_df["TOTAL"].rolling(
            window=short_window).mean()
        long_mean_total = log_returns_df["TOTAL"].rolling(
            window=long_window).mean()
        spread_total = short_mean_total - long_mean_total
        spread_mean_total = spread_total.rolling(window=long_window).mean()
        spread_std_total = spread_total.rolling(window=long_window).std()
        spread_zscore["TOTAL"] = (
            spread_total - spread_mean_total) / spread_std_total
        return spread_zscore

    def get_volatility_signal(self, log_returns_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """
        Calculate rolling volatility (std dev) of log returns for each symbol and TOTAL.
        Args:
            log_returns_df: DataFrame of log returns (should include all symbols and TOTAL)
            window: Rolling window size (default 63)
        Returns:
            DataFrame of rolling std dev for each symbol and TOTAL
        """
        vol = log_returns_df[self.valid_symbols].rolling(window=window).std()
        vol_total = log_returns_df["TOTAL"].rolling(window=window).std()
        vol["TOTAL"] = vol_total
        return vol

    def get_sharpe_signal(self, log_returns_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """
        Calculate rolling Sharpe ratio (mean/std) of log returns for each symbol and TOTAL.
        Args:
            log_returns_df: DataFrame of log returns (should include all symbols and TOTAL)
            window: Rolling window size (default 63)
        Returns:
            DataFrame of rolling Sharpe ratio for each symbol and TOTAL
        """
        mean = log_returns_df[self.valid_symbols].rolling(window=window).mean()
        std = log_returns_df[self.valid_symbols].rolling(window=window).std()
        sharpe = mean / std
        mean_total = log_returns_df["TOTAL"].rolling(window=window).mean()
        std_total = log_returns_df["TOTAL"].rolling(window=window).std()
        sharpe_total = mean_total / std_total
        sharpe["TOTAL"] = sharpe_total
        return sharpe

    def get_relative_strength_signal(self, price_df: pd.DataFrame, total_b_value: pd.Series, window: int = 63) -> pd.DataFrame:
        """
        Calculate relative strength: rolling window return of each symbol minus TOTAL.
        Args:
            price_df: DataFrame of prices (should include all symbols and TOTAL)
            window: Rolling window size (default 63)
        Returns:
            DataFrame of relative strength for each symbol and TOTAL (TOTAL is NaN)
        """
        returns = price_df[self.valid_symbols].pct_change(periods=window)
        total_return = total_b_value.pct_change(periods=window)
        rel_strength = returns.subtract(total_return, axis=0)
        rel_strength["TOTAL"] = np.nan
        return rel_strength

    def get_simple_strategy_weights(self, price_df: pd.DataFrame, benchmark_weights: pd.DataFrame, window: int = 21) -> pd.DataFrame:
        """
        Simple momentum strategy: tilt away from benchmark weights based on recent returns.

        Args:
            price_df: DataFrame of prices
            benchmark_weights: DataFrame of current benchmark weights
            window: Rolling window for momentum calculation (default 21)

        Returns:
            DataFrame of portfolio weights
        """
        # Calculate momentum (rolling return over the window)
        momentum = price_df[self.valid_symbols].pct_change(periods=window)
        momentum_adjustment = momentum.fillna(0) * 0.3  # 30% tilt factor

        # Use current benchmark weights as base
        portfolio_weights = benchmark_weights[self.valid_symbols] + \
            momentum_adjustment

        # Ensure weights sum to 1 and are non-negative
        portfolio_weights = portfolio_weights.clip(lower=0)
        portfolio_weights = portfolio_weights.div(
            portfolio_weights.sum(axis=1), axis=0)
        return portfolio_weights

    def _validate_return_calculations(self, value_series: pd.Series, log_return_series: pd.Series, prefix: str) -> None:
        """
        Validate that return calculations are mathematically correct and print the result.
        Also print CAGR, annualized log return mean, and stddev.

        Args:
            value_series: Series of portfolio values (for a symbol or TOTAL)
            log_return_series: Series of log returns (for a symbol or TOTAL)
            prefix: Prefix for print messages
        """
        first_value = value_series.iloc[0]
        last_value = value_series.iloc[-1]
        exp_sum_log_returns = np.exp(
            log_return_series[1:].sum())  # skip first NaN
        test_last_value = first_value * exp_sum_log_returns

        # Fix date arithmetic
        start_date = pd.to_datetime(value_series.index[0])
        end_date = pd.to_datetime(value_series.index[-1])
        n_years = (end_date - start_date).days / 365.25

        cagr = (last_value / first_value) ** (1 / n_years) - \
            1 if n_years > 0 else float('nan')
        ann_log_return_mean = log_return_series[1:].mean() * 252
        ann_log_return_std = log_return_series[1:].std() * (252 ** 0.5)
        print(f"{prefix} validation:")
        print(f"  Sample years: {n_years:.2f}")
        print(f"  First Value: {first_value:.4f}")
        print(f"  exp(sum(log_returns)): {exp_sum_log_returns:.6f}")
        print(f"  Test Last Value: {test_last_value:.4f}")
        print(f"  Actual Last Value: {last_value:.4f}")
        print(f"  CAGR: {cagr:.4%}")
        print(f"  Annualized log return mean: {ann_log_return_mean:.4%}")
        print(f"  Annualized log return stddev: {ann_log_return_std:.4%}")
        if not np.isclose(test_last_value, last_value, atol=1e-4):
            print("  WARNING: Validation failed! Values do not match.")
            raise ValueError("Validation failed! Values do not match.")
        else:
            print("  Validation successful: Values match.\n")

    def get_main_dataframes(self) -> pd.DataFrame:
        """
        Get the multi-index DataFrame containing all main data components.

        Returns:
            Multi-index DataFrame with levels: ["Metric", "Ticker"]
            Contains: Price, B_Position, B_Value, B_Weight, B_log_returns (plus TOTAL column)
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")
        return self.main_data_multiindex

    def get_tickers_history(self) -> pd.DataFrame:
        """
        Get the historical price data for all tickers.

        Returns:
            DataFrame with historical close prices
        """
        if not isinstance(self.tickers_data, dict):
            raise ValueError("Invalid tickers data format")
        return self.tickers_data["history"]

    def get_tickers_news(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get news data for all tickers.

        Returns:
            Dictionary mapping ticker symbols to news articles
        """
        if not isinstance(self.tickers_data, dict):
            raise ValueError("Invalid tickers data format")
        return self.tickers_data["news"]

    def get_tickers_data(self) -> Union[str, TickersData]:
        """
        Get the complete tickers data.

        Returns:
            Either error string or complete tickers data dictionary
        """
        if self.tickers_data is None:
            raise ValueError("Data not yet fetched. Call _fetch_data() first.")
        return self.tickers_data

    def get_portfolio_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get all portfolio-related DataFrames for analysis.

        Returns:
            Dictionary containing:
            - p_position_df: Portfolio positions
            - p_value_df: Portfolio values  
            - p_weight_df: Portfolio weights
            - p_log_returns_df: Portfolio log returns
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        return {
            'p_position_df': self.main_data_multiindex['P_Position'],
            'p_value_df': self.main_data_multiindex['P_Value'],
            'p_weight_df': self.main_data_multiindex['P_Weight'],
            'p_log_returns_df': self.main_data_multiindex['P_log_returns']
        }

    def get_benchmark_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Get all benchmark-related DataFrames for analysis.

        Returns:
            Dictionary containing:
            - b_position_df: Benchmark positions
            - b_value_df: Benchmark values  
            - b_weight_df: Benchmark weights
            - b_log_returns_df: Benchmark log returns
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        return {
            'b_position_df': self.main_data_multiindex['B_Position'],
            'b_value_df': self.main_data_multiindex['B_Value'],
            'b_weight_df': self.main_data_multiindex['B_Weight'],
            'b_log_returns_df': self.main_data_multiindex['B_log_returns']
        }

    def compare_performance(self) -> Dict[str, float]:
        """
        Compare portfolio vs benchmark performance.

        Returns:
            Dictionary with performance metrics:
            - benchmark_total_return: Cumulative benchmark return
            - portfolio_total_return: Cumulative portfolio return
            - outperformance: Portfolio minus benchmark return
            - benchmark_annualized: Annualized benchmark return
            - portfolio_annualized: Annualized portfolio return
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        # Get cumulative returns
        benchmark_cumulative = self.main_data_multiindex[(
            "B_log_returns", "TOTAL")].sum()
        portfolio_cumulative = self.main_data_multiindex[(
            "P_log_returns", "TOTAL")].sum()

        # Calculate time period
        dates = self.main_data_multiindex.index
        n_years = (dates[-1] - dates[0]).days / 365.25

        # Annualize returns
        benchmark_annualized = np.exp(
            benchmark_cumulative / n_years) - 1 if n_years > 0 else 0
        portfolio_annualized = np.exp(
            portfolio_cumulative / n_years) - 1 if n_years > 0 else 0

        return {
            'benchmark_total_return': np.exp(benchmark_cumulative) - 1,
            'portfolio_total_return': np.exp(portfolio_cumulative) - 1,
            'outperformance': np.exp(portfolio_cumulative) - np.exp(benchmark_cumulative),
            'benchmark_annualized': benchmark_annualized,
            'portfolio_annualized': portfolio_annualized,
            'period_years': n_years
        }

    def get_backtest_results(self) -> BacktestResults:
        """
        Get comprehensive backtest results.

        Returns:
            BacktestResults object containing all performance metrics and data
        """
        if self.main_data_multiindex is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        # Get benchmark and portfolio performance from TOTAL columns
        benchmark_perf = self.main_data_multiindex[(
            "B_log_returns", "TOTAL")].sum()
        portfolio_perf = self.main_data_multiindex[(
            "P_log_returns", "TOTAL")].sum()
        outperformance = portfolio_perf - benchmark_perf

        return BacktestResults(
            benchmark_performance=benchmark_perf,
            portfolio_performance=portfolio_perf,
            outperformance=outperformance,
            consolidated_data=pd.DataFrame(),
            main_data_multiindex=self.main_data_multiindex,
            tickers_history=self.get_tickers_history(),
            tickers_news=self.get_tickers_news()
        )
