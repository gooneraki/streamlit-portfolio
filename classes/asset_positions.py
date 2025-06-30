""" Module for asset positions and portfolio """
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utilities.app_yfinance import tickers_yf, yf_ticket_info


@dataclass
class AssetPosition:
    """ Asset position """
    symbol: str
    position: float

    def get_symbol(self) -> str:
        """ Get the symbol """
        return self.symbol

    def get_position(self) -> float:
        """ Get the position """
        return self.position


class Portfolio:
    """ Portfolio class """

    def __init__(self, asset_positions: list[AssetPosition], reference_currency: str):
        positions_series = pd.Series(
            data=[position.get_position() for position in asset_positions],
            index=[position.get_symbol() for position in asset_positions],
        )

        asset_symbols = [position.get_symbol() for position in asset_positions]

        reference_currency = reference_currency.upper()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.symbols_info_list = [yf_ticket_info(
            symbol) for symbol in asset_symbols]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        currencies_series = pd.Series(
            data=[info['currency'] for info in self.symbols_info_list],
            index=[position.get_symbol() for position in asset_positions],
        )

        currency_symbols = [(target_currency + reference_currency + "=X")
                            for target_currency in currencies_series.unique() if target_currency != reference_currency]

        # >>>
        self.tickers_data = tickers_yf(
            asset_symbols+currency_symbols, period='max')
        # <<<

        translated_values = self._get_translated_history(
            asset_positions, reference_currency, currency_symbols)

        weights = translated_values.div(translated_values.sum(axis=1), axis=0)

        translated_values['TOTAL'] = translated_values.sum(axis=1)

        translated_fitted_values, trend_deviation, trend_deviation_z_score = self._get_fitted_values(
            translated_values)

        _, _, _, _, number_of_years, _, _ = self.get_period_info()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.timeseries_data = pd.concat(
            objs=[
                weights,
                translated_values, translated_fitted_values,
                trend_deviation, trend_deviation_z_score],
            axis=1,
            keys=[
                'weights',
                'translated_values', 'translated_fitted_values',
                'trend_deviation', 'trend_deviation_z_score'])

        self.timeseries_data.columns.names = ['Metric', 'Ticker']
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        latest_timeseries_data: pd.Series = self.timeseries_data.iloc[-1]
        latest_timeseries_df: pd.DataFrame = latest_timeseries_data.unstack(
            level='Metric')

        first_value, last_value, first_fitted_value, last_fitted_value = self.get_first_last_values(
            translated_values, translated_fitted_values)

        cagr: pd.Series = (
            last_value / first_value) ** (1 / number_of_years) - 1
        cagr_fitted: pd.Series = (last_fitted_value /
                                  first_fitted_value) ** (1 / number_of_years) - 1

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.assets_metrics = pd.concat(
            objs=[
                currencies_series, positions_series,
                latest_timeseries_df['weights'], latest_timeseries_df['weights'].mul(
                    100),
                cagr, cagr_fitted,
                cagr.mul(100), cagr_fitted.mul(100),
                last_value, last_fitted_value,
                latest_timeseries_df['trend_deviation'],
                latest_timeseries_df['trend_deviation_z_score']],
            axis=1,
            keys=[
                'currency', 'position',
                'latest_weights', 'latest_weights_pct',
                'cagr', 'cagr_fitted',
                'cagr_pct', 'cagr_fitted_pct',
                'latest_value', 'latest_fitted_value',
                'trend_deviation', 'trend_deviation_z_score'])
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Calculate both optimal weights and efficient frontier data
        self.optimal_weights, self.efficient_frontier_data = self._calculate_portfolio_optimization()
        print(f"\n🎯 FINAL OPTIMAL WEIGHTS RESULT:\n{self.optimal_weights}\n")
        # print(
        #     f"\n📈 EFFICIENT FRONTIER DATA:\n{self.efficient_frontier_data}\n")

    def get_assets_metrics(self):
        """ Get the assets metrics """
        return self.assets_metrics

    def get_optimal_weights(self):
        """ Get the optimal weights from efficient frontier calculations """
        return self.optimal_weights

    def get_efficient_frontier_data(self):
        """ Get the efficient frontier data points for plotting """
        return self.efficient_frontier_data

    def get_asset_series(self, p_asset: str):
        """ Get the series for the given asset """

        if not p_asset in self.timeseries_data.columns.get_level_values('Ticker').unique():
            raise ValueError(f"Asset {p_asset} not found in timeseries data")

        return self.timeseries_data.xs(p_asset, level='Ticker', axis=1)

    def get_first_last_values(self, p_history: pd.DataFrame, p_fitted_history: pd.DataFrame):
        """ Get the first and last values """
        first_value: pd.Series = p_history.iloc[0]
        last_value: pd.Series = p_history.iloc[-1]

        first_fitted_value: pd.Series = p_fitted_history.iloc[0]
        last_fitted_value: pd.Series = p_fitted_history.iloc[-1]

        if not isinstance(first_value, pd.Series):
            raise ValueError("First value is not a Series")

        if not isinstance(last_value, pd.Series):
            raise ValueError("Last value is not a Series")

        if not isinstance(first_fitted_value, pd.Series):
            raise ValueError("First fitted value is not a Series")

        if not isinstance(last_fitted_value, pd.Series):
            raise ValueError("Last fitted value is not a Series")

        return first_value, last_value, first_fitted_value, last_fitted_value

    def get_period_info(self):
        """ Get the period info for the given history """
        first_date = self.tickers_data['history'].index[0]
        last_date = self.tickers_data['history'].index[-1]

        if not isinstance(first_date, pd.Timestamp):
            raise ValueError("First date is not a Timestamp")
        if not isinstance(last_date, pd.Timestamp):
            raise ValueError("Last date is not a Timestamp")

        number_of_points = self.tickers_data['history'].shape[0]
        number_of_days = (last_date - first_date).days + 1
        number_of_years = number_of_days / 365.25
        points_per_year = number_of_points / number_of_years
        points_per_month = points_per_year / 12

        return first_date, last_date, number_of_points, number_of_days, \
            number_of_years, points_per_year, points_per_month

    def _get_fitted_values(self, translated_values: pd.DataFrame):
        """ Get fitted values for each symbol individually """

        fitted_values = pd.DataFrame(index=translated_values.index)

        for column in translated_values.columns:
            x = np.arange(len(translated_values))
            y_log = np.log(translated_values[column])
            z_exp = np.polyfit(x, y_log, 1)
            p_exp = np.poly1d(z_exp)
            y_exp = np.exp(p_exp(x))

            fitted_values[column] = y_exp

        trend_deviation = (
            translated_values / fitted_values) - 1

        if not isinstance(trend_deviation, pd.DataFrame):
            raise ValueError("Trend deviation is not a DataFrame")

        trend_deviation_z_score = (trend_deviation - trend_deviation.mean()) / \
            trend_deviation.std()

        if not isinstance(trend_deviation_z_score, pd.DataFrame):
            raise ValueError("Trend deviation z-score is not a DataFrame")

        return fitted_values, trend_deviation, trend_deviation_z_score

    def _get_fitted_total_values(self, translated_total: pd.Series):
        """ Get fitted values for the total portfolio """

        x = np.arange(len(translated_total))
        y_log = np.log(translated_total)
        z_exp = np.polyfit(x, y_log, 1)
        p_exp = np.poly1d(z_exp)
        y_exp = np.exp(p_exp(x))

        fitted_total = pd.Series(
            y_exp, index=translated_total.index, name='fitted_total')
        return fitted_total

    def _get_translated_history(
            self,
            asset_positions: list[AssetPosition],
            reference_currency: str,
            currency_symbols: list[str]):

        assets_history = self.tickers_data['history'].drop(
            columns=currency_symbols)
        currency_history = self.tickers_data['history'][currency_symbols]

        if not isinstance(currency_history, pd.DataFrame):
            raise ValueError("Currency history is not a DataFrame")

        translated_values = assets_history.copy()

        # Create a dictionary to map asset_symbols to positions for quick lookup
        positions_dict = {pos.get_symbol(): pos.get_position()
                          for pos in asset_positions}

        for symbol, info in zip(assets_history.columns, self.symbols_info_list):
            position_value = positions_dict[symbol]

            translated_values[symbol] = translated_values[symbol] * \
                position_value

            if info['currency'] != reference_currency:
                currency_column = info['currency'] + reference_currency + "=X"
                translated_values[symbol] = translated_values[symbol] * \
                    currency_history[currency_column]

        return translated_values

    def _calculate_portfolio_optimization(self, num_points=20):
        """ Calculate both optimal weights and efficient frontier data """

        # Get asset data (excluding TOTAL column and currency conversion symbols)
        asset_data = self.timeseries_data.xs(
            'translated_values', level='Metric', axis=1)
        asset_data = asset_data.drop(columns=['TOTAL'], errors='ignore')

        # Calculate returns
        returns = asset_data.pct_change().dropna()
        expected_returns = returns.mean()
        if not isinstance(expected_returns, pd.Series):
            raise ValueError("Expected returns is not a Series")

        cov_matrix = returns.cov()
        n_assets = len(asset_data.columns)

        # Common optimization setup
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0/n_assets] * n_assets)

        # ================================
        # PART 1: OPTIMAL WEIGHTS
        # ================================

        optimal_weights_dict = {}

        # Maximum Sharpe Ratio Portfolio
        def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return -
                            risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # We minimize, so return negative

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Optimize for maximum Sharpe ratio

        result = minimize(neg_sharpe_ratio, initial_guess,
                          args=(expected_returns, cov_matrix),
                          method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:

            optimal_weights_dict['max_sharpe'] = pd.Series(
                result.x, index=expected_returns.index)

        else:
            print("❌ Max Sharpe optimization failed, using equal weights")
            optimal_weights_dict['max_sharpe'] = pd.Series(
                [1.0/n_assets] * n_assets, index=expected_returns.index)

        # ================================
        # PART 2: EFFICIENT FRONTIER
        # ================================

        # Define portfolio variance function for efficient frontier

        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Create target returns range
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_points)

        frontier_data = []

        for target_return in target_returns:
            # Constraints for efficient frontier
            constraints_ef = [
                {'type': 'eq', 'fun': lambda x: np.sum(
                    x) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda x, ret=target_return: np.sum(
                    x * expected_returns) - ret}  # target return
            ]

            # Optimize
            result = minimize(portfolio_variance, initial_guess,
                              args=(cov_matrix,),
                              method='SLSQP', bounds=bounds, constraints=constraints_ef)

            if result.success:
                portfolio_return = np.sum(result.x * expected_returns)
                portfolio_volatility = np.sqrt(
                    np.dot(result.x.T, np.dot(cov_matrix, result.x)))

                frontier_data.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': (portfolio_return - 0.02) / portfolio_volatility,
                    'weights': pd.Series(result.x, index=expected_returns.index)
                })

        if frontier_data:
            efficient_frontier_df = pd.DataFrame(frontier_data)
        else:
            print("❌ No successful optimizations for efficient frontier")
            efficient_frontier_df = pd.DataFrame()

        return optimal_weights_dict, efficient_frontier_df
