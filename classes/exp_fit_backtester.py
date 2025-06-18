import pandas as pd
import numpy as np
from utilities.app_yfinance import tickers_yf
from typing import Literal
from utilities.utilities import get_rolling_exp_fit


def print_df(df, title):
    print(f"\n{title}")
    print(df.head())
    print(df.tail())
    print(df.shape)


def get_central_deviations(close_data):

    tickers_history_fitted = close_data.apply(
        get_rolling_exp_fit, min_points=round(365.25*3))

    deviations = (tickers_history_fitted/close_data - 1).dropna()
    central_deviations = deviations.sub(deviations.mean(axis=1), axis=0)

    return central_deviations


def get_portfolio_weights(benchmark_weights, central_deviations, tilt, period: Literal['weekly', 'monthly']):

    # print_df(benchmark_weights, "Benchmark Weights")
    # (1759, 11)
    # print_df(central_deviations, "Central Deviations")
    # (1740, 11)

    # WEIGHT BALANCING HAPPENS HERE
    portfolio_weights_pre = (benchmark_weights +
                             tilt * central_deviations).dropna()

    # print_df(portfolio_weights_pre, "Portfolio Weights Pre")
    # (1740, 11)

    weights_last_period = portfolio_weights_pre.resample(
        'W' if period == 'weekly' else 'ME').last()

    # print_df(weights_last_period, "Weights Last Period")
    # (84, 11)

    weights_reindexed = weights_last_period.reindex(
        central_deviations.index, method='ffill').dropna()
    # print_df(weights_reindexed, "Weights Reindexed")
    # (1730, 11)

    # # Align the data - use portfolio_weights where available, benchmark_weights as fallback
    portfolio_weights = benchmark_weights.copy()
    portfolio_weights.loc[weights_reindexed.index,
                          weights_reindexed.columns] = weights_reindexed

    return portfolio_weights


class ExpFitBacktester():
    column_titles = ["Metric", "Ticker"]
    tilt = 0.5

    def __init__(self, symbols: list[str]):
        self.symbols = symbols

        self.fetch_data()

    def fetch_data(self):
        self.tickers_data = tickers_yf(self.symbols)
        if isinstance(self.tickers_data, str):
            raise ValueError(self.tickers_data)

        self.preprocess_data()

    def preprocess_data(self):
        # Remove full null columns (i.e. symbols with no data)
        self.tickers_data["history"] = self.tickers_data["history"].dropna(
            axis=1, how='all')
        # Remove null rows (i.e. keep dates with all symbols having data)
        self.tickers_data["history"] = self.tickers_data["history"].dropna(
            axis=0, how='any')

        # Get valid symbols
        self.valid_symbols = self.tickers_data["history"].columns.to_list()

        if len(self.valid_symbols) == 0:
            raise ValueError("No valid symbols found")

        close_data = self.tickers_data["history"]
        benchmark_weights = pd.DataFrame(
            1 / len(self.valid_symbols),
            index=close_data.index,
            columns=close_data.columns,
        )
        benchmark_close_data = (close_data * benchmark_weights).sum(axis=1)

        #####
        # Calculate benchmark-level returns
        # Lost 1 row because of the log_returns
        benchmark_log_returns = np.log(
            benchmark_close_data / benchmark_close_data.shift(1)).dropna()
        benchmark_cum_returns = benchmark_log_returns.cumsum()
        benchmark_exp_cum_returns = np.exp(benchmark_cum_returns)

        # Refer to this to understand the math
        test_total_close = benchmark_close_data.iloc[0] * \
            benchmark_exp_cum_returns.iloc[-1]
        actual_total_close = benchmark_close_data.iloc[-1]

        if round(test_total_close - actual_total_close, 4) != 0:
            raise ValueError(
                f"Test Total Close: {test_total_close} != Actual Total Close: {actual_total_close}")
        # Lost end
        #####

        #####
        # Returns per ticker
        # Lost 1 row because of the log_returns
        # log_returns = close_data.apply(
        #     lambda x: np.log(x / x.shift(1))).dropna()
        # cum_returns = log_returns.cumsum()
        # exp_cum_returns = np.exp(cum_returns)
        # lost end
        #####

        central_deviations = get_central_deviations(close_data)

        # WEIGHT BALANCING HAPPENS HERE
        portfolio_weights = get_portfolio_weights(
            benchmark_weights, central_deviations, self.tilt, 'monthly')
        # print_df(portfolio_weights, "Portfolio Weights")

        portfolio_close_data = (close_data * portfolio_weights).sum(axis=1)

        #####
        # Calculate portfolio returns
        # Lost 1 row because of the log_returns
        portfolio_log_returns = np.log(
            portfolio_close_data / portfolio_close_data.shift(1)).dropna()

        portfolio_cum_returns = portfolio_log_returns.cumsum()
        portfolio_exp_cum_returns = np.exp(portfolio_cum_returns)

        # Refer to this to understand the math
        portfolio_total_close: float = (portfolio_close_data.iloc[0] *
                                        portfolio_exp_cum_returns).iloc[-1]
        actual_portfolio_total_close: float = portfolio_close_data.iloc[-1]

        if round(portfolio_total_close - actual_portfolio_total_close, 4) != 0:
            raise ValueError(
                f"Test Portfolio Total Close: {portfolio_total_close} != Actual Portfolio Total Close: {actual_portfolio_total_close}")
        # lost end
        #####

        # Print benchmark return and
        print(f"Benchmark Performance: {benchmark_exp_cum_returns.iloc[-1]}")
        print(f"Portfolio Performance: {portfolio_exp_cum_returns.iloc[-1]}")
        print(
            f"Outperformance: {portfolio_exp_cum_returns.iloc[-1] - benchmark_exp_cum_returns.iloc[-1]}")

        print_df(close_data, "Close Data")  # (1760, 11)
        print_df(benchmark_weights, "Benchmark Weights")  # (1760, 11)
        print_df(central_deviations, "Central Deviations")  # (1760, 11)
        print_df(portfolio_weights, "Portfolio Weights")  # (1760, 11)

        # Create concatenated dataframe with all four dataframes
        # Add prefixes to distinguish between different dataframes
        close_data_prefixed = close_data.add_prefix('close_')
        benchmark_weights_prefixed = benchmark_weights.add_prefix(
            'benchmark_weights_')
        central_deviations_prefixed = central_deviations.add_prefix(
            'central_deviations_')
        portfolio_weights_prefixed = portfolio_weights.add_prefix(
            'portfolio_weights_')

        # Concatenate all dataframes horizontally
        self.concatenated_main_data = pd.concat([
            close_data_prefixed,
            benchmark_weights_prefixed,
            central_deviations_prefixed,
            portfolio_weights_prefixed
        ], axis=1)

        # Create multi-index dataframe with the four main dataframes
        # Create a list of tuples for the multi-index columns
        multi_index_columns = []

        # Add close_data columns
        for ticker in close_data.columns:
            multi_index_columns.append(('close_data', ticker))

        # Add benchmark_weights columns
        for ticker in benchmark_weights.columns:
            multi_index_columns.append(('benchmark_weights', ticker))

        # Add central_deviations columns
        for ticker in central_deviations.columns:
            multi_index_columns.append(('central_deviations', ticker))

        # Add portfolio_weights columns
        for ticker in portfolio_weights.columns:
            multi_index_columns.append(('portfolio_weights', ticker))

        # Create the multi-index
        multi_index = pd.MultiIndex.from_tuples(
            multi_index_columns, names=self.column_titles)

        # Concatenate all dataframes and assign the multi-index
        self.main_data_multiindex = pd.concat([
            close_data,
            benchmark_weights,
            central_deviations,
            portfolio_weights
        ], axis=1)
        self.main_data_multiindex.columns = multi_index

        print_df(benchmark_close_data, "Benchmark Close Data")  # (1760,)
        print_df(benchmark_log_returns, "Benchmark Log Returns")  # (1759,)
        print_df(benchmark_cum_returns, "Benchmark Cum Returns")  # (1759,)
        print_df(benchmark_exp_cum_returns,
                 "Benchmark Exp Cum Returns")  # (1759,)

        print_df(portfolio_close_data, "Portfolio Close Data")  # (1760,)
        print_df(portfolio_log_returns, "Portfolio Log Returns")  # (1759,)
        print_df(portfolio_cum_returns, "Portfolio Cum Returns")  # (1759,)
        print_df(portfolio_exp_cum_returns,
                 "Portfolio Exp Cum Returns")  # (1759,)

        data_dict = {
            'benchmark_close': benchmark_close_data,
            'benchmark_log_returns': benchmark_log_returns,
            'benchmark_cum_returns': benchmark_cum_returns,
            'benchmark_exp_cum_returns': benchmark_exp_cum_returns,
            'portfolio_close': portfolio_close_data,
            'portfolio_log_returns': portfolio_log_returns,
            'portfolio_cum_returns': portfolio_cum_returns,
            'portfolio_exp_cum_returns': portfolio_exp_cum_returns
        }

        # Create consolidated dataframe with all metrics
        self.consolidated_data = pd.DataFrame(data_dict)

    def get_consolidated_data(self):
        """
        Return the consolidated dataframe with all benchmark and portfolio metrics.
        """
        return self.consolidated_data

    def get_main_dataframes(self):
        """
        Return the multi-index dataframe containing all four main dataframes:
        - close_data (with 'close_data' as Metric level)
        - benchmark_weights (with 'benchmark_weights' as Metric level)
        - central_deviations (with 'central_deviations' as Metric level)  
        - portfolio_weights (with 'portfolio_weights' as Metric level)

        The dataframe has a MultiIndex with levels: ["Metric", "Ticker"]
        """
        return self.main_data_multiindex

    def get_tickers_history(self):
        return self.tickers_data["history"]

    def get_tickers_news(self):
        return self.tickers_data["news"]

    def get_tickers_data(self):
        return self.tickers_data
