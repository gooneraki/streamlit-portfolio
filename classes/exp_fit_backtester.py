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
from utilities.utilities import get_rolling_exp_fit


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
    min_points = round(365.25 * 3)
    tickers_history_fitted = close_data.apply(
        get_rolling_exp_fit, min_points=min_points
    )

    # Calculate deviations from fitted values
    deviations = (tickers_history_fitted / close_data - 1).dropna()

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

    def __init__(self, symbols: List[str]):
        """
        Initialize the backtester with a list of symbols.

        Args:
            symbols: List of ticker symbols to include in the backtest

        Raises:
            ValueError: If no valid symbols are found or data fetching fails
        """
        self.symbols = symbols
        self.tickers_data: Optional[Union[str, TickersData]] = None
        self.valid_symbols: List[str] = []
        self.consolidated_data: Optional[pd.DataFrame] = None
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

        This method performs the core backtesting calculations including:
        - Benchmark and portfolio returns
        - Central deviations from exponential fits
        - Portfolio weight optimization
        - Performance comparison
        """
        if not isinstance(self.tickers_data, dict):
            raise ValueError("Invalid tickers data format")

        close_data = self.tickers_data["history"]

        # Create equal-weighted benchmark
        benchmark_weights = pd.DataFrame(
            1 / len(self.valid_symbols),
            index=close_data.index,
            columns=close_data.columns,
        )

        # Calculate benchmark performance
        benchmark_close_data = (close_data * benchmark_weights).sum(axis=1)
        benchmark_metrics = self._calculate_return_metrics(
            benchmark_close_data, "benchmark")

        # Calculate central deviations and portfolio weights
        central_deviations = get_central_deviations(close_data)
        portfolio_weights = get_portfolio_weights(
            benchmark_weights, central_deviations, self.tilt, 'monthly'
        )

        # Calculate portfolio performance
        portfolio_close_data = (close_data * portfolio_weights).sum(axis=1)
        portfolio_metrics = self._calculate_return_metrics(
            portfolio_close_data, "portfolio")

        # Print performance summary
        self._print_performance_summary(benchmark_metrics, portfolio_metrics)

        # Create consolidated data structures
        self._create_data_structures(
            close_data, benchmark_weights, central_deviations, portfolio_weights,
            benchmark_metrics, portfolio_metrics
        )

    def _calculate_return_metrics(
        self, close_data: pd.Series, prefix: str
    ) -> Dict[str, pd.Series]:
        """
        Calculate return metrics for a given price series.

        Args:
            close_data: Series of close prices
            prefix: Prefix for metric names

        Returns:
            Dictionary containing log returns, cumulative returns, and exponential cumulative returns
        """
        # Calculate log returns
        log_returns = np.log(close_data / close_data.shift(1)).dropna()
        cum_returns = log_returns.cumsum()
        exp_cum_returns = np.exp(cum_returns)

        # Validate calculations
        self._validate_return_calculations(close_data, exp_cum_returns, prefix)

        return {
            'log_returns': log_returns,
            'cum_returns': cum_returns,
            'exp_cum_returns': exp_cum_returns
        }

    def _validate_return_calculations(
        self, close_data: pd.Series, exp_cum_returns: pd.Series, prefix: str
    ) -> None:
        """
        Validate that return calculations are mathematically correct.

        Args:
            close_data: Original close price series
            exp_cum_returns: Exponential cumulative returns
            prefix: Prefix for error messages

        Raises:
            ValueError: If calculations are incorrect
        """
        test_total_close = close_data.iloc[0] * exp_cum_returns.iloc[-1]
        actual_total_close = close_data.iloc[-1]

        if round(test_total_close - actual_total_close, 4) != 0:
            raise ValueError(
                f"{prefix.capitalize()} calculation error: "
                f"Test Total Close: {test_total_close} != "
                f"Actual Total Close: {actual_total_close}"
            )

    def _print_performance_summary(
        self, benchmark_metrics: Dict[str, pd.Series],
        portfolio_metrics: Dict[str, pd.Series]
    ) -> None:
        """
        Print a summary of benchmark vs portfolio performance.

        Args:
            benchmark_metrics: Benchmark return metrics
            portfolio_metrics: Portfolio return metrics
        """
        benchmark_perf = benchmark_metrics['exp_cum_returns'].iloc[-1]
        portfolio_perf = portfolio_metrics['exp_cum_returns'].iloc[-1]
        outperformance = portfolio_perf - benchmark_perf

        print(f"Benchmark Performance: {benchmark_perf:.4f}")
        print(f"Portfolio Performance: {portfolio_perf:.4f}")
        print(f"Outperformance: {outperformance:.4f}")

    def _create_data_structures(
        self,
        close_data: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        central_deviations: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        benchmark_metrics: Dict[str, pd.Series],
        portfolio_metrics: Dict[str, pd.Series]
    ) -> None:
        """
        Create consolidated data structures for analysis.

        Args:
            close_data: Historical close prices
            benchmark_weights: Benchmark portfolio weights
            central_deviations: Central deviations from exponential fits
            portfolio_weights: Optimized portfolio weights
            benchmark_metrics: Benchmark return metrics
            portfolio_metrics: Portfolio return metrics
        """
        # Create multi-index DataFrame
        self._create_multiindex_dataframe(
            close_data, benchmark_weights, central_deviations, portfolio_weights
        )

        # Create consolidated DataFrame with all metrics
        self._create_consolidated_dataframe(
            benchmark_metrics, portfolio_metrics)

    def _create_multiindex_dataframe(
        self,
        close_data: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        central_deviations: pd.DataFrame,
        portfolio_weights: pd.DataFrame
    ) -> None:
        """
        Create a multi-index DataFrame with all main data components.

        Args:
            close_data: Historical close prices
            benchmark_weights: Benchmark portfolio weights
            central_deviations: Central deviations from exponential fits
            portfolio_weights: Optimized portfolio weights
        """
        # Create multi-index columns
        multi_index_columns = []

        for ticker in close_data.columns:
            multi_index_columns.extend([
                ('close_data', ticker),
                ('benchmark_weights', ticker),
                ('central_deviations', ticker),
                ('portfolio_weights', ticker)
            ])

        multi_index = pd.MultiIndex.from_tuples(
            multi_index_columns, names=self.column_titles
        )

        # Concatenate all dataframes
        self.main_data_multiindex = pd.concat([
            close_data, benchmark_weights, central_deviations, portfolio_weights
        ], axis=1)
        self.main_data_multiindex.columns = multi_index

    def _create_consolidated_dataframe(
        self,
        benchmark_metrics: Dict[str, pd.Series],
        portfolio_metrics: Dict[str, pd.Series]
    ) -> None:
        """
        Create a consolidated DataFrame with all performance metrics.

        Args:
            benchmark_metrics: Benchmark return metrics
            portfolio_metrics: Portfolio return metrics
        """
        data_dict = {
            'benchmark_close': benchmark_metrics.get('close_data', pd.Series()),
            'benchmark_log_returns': benchmark_metrics['log_returns'],
            'benchmark_cum_returns': benchmark_metrics['cum_returns'],
            'benchmark_exp_cum_returns': benchmark_metrics['exp_cum_returns'],
            'portfolio_close': portfolio_metrics.get('close_data', pd.Series()),
            'portfolio_log_returns': portfolio_metrics['log_returns'],
            'portfolio_cum_returns': portfolio_metrics['cum_returns'],
            'portfolio_exp_cum_returns': portfolio_metrics['exp_cum_returns']
        }

        self.consolidated_data = pd.DataFrame(data_dict)

    def get_consolidated_data(self) -> pd.DataFrame:
        """
        Get the consolidated DataFrame with all benchmark and portfolio metrics.

        Returns:
            DataFrame containing all performance metrics
        """
        if self.consolidated_data is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")
        return self.consolidated_data

    def get_main_dataframes(self) -> pd.DataFrame:
        """
        Get the multi-index DataFrame containing all main data components.

        Returns:
            Multi-index DataFrame with levels: ["Metric", "Ticker"]
            Contains: close_data, benchmark_weights, central_deviations, portfolio_weights
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

    def get_backtest_results(self) -> BacktestResults:
        """
        Get comprehensive backtest results.

        Returns:
            BacktestResults object containing all performance metrics and data
        """
        if self.consolidated_data is None:
            raise ValueError(
                "Data not yet processed. Call _preprocess_data() first.")

        if self.main_data_multiindex is None:
            raise ValueError("Multi-index data not yet created.")

        benchmark_perf = self.consolidated_data['benchmark_exp_cum_returns'].iloc[-1]
        portfolio_perf = self.consolidated_data['portfolio_exp_cum_returns'].iloc[-1]
        outperformance = portfolio_perf - benchmark_perf

        return BacktestResults(
            benchmark_performance=benchmark_perf,
            portfolio_performance=portfolio_perf,
            outperformance=outperformance,
            consolidated_data=self.consolidated_data,
            main_data_multiindex=self.main_data_multiindex,
            tickers_history=self.get_tickers_history(),
            tickers_news=self.get_tickers_news()
        )
