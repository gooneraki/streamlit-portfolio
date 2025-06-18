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

    tickers_history_fitted = close_data.apply(get_rolling_exp_fit)

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

        self.tickers_data = None
        self.valid_symbols = None

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

    def get_tickers_history(self):
        return self.tickers_data["history"]

    def get_tickers_news(self):
        return self.tickers_data["news"]

    def get_tickers_data(self):
        return self.tickers_data
