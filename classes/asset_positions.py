""" Module for asset positions and portfolio """

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.core.window.rolling import Rolling
from scipy.optimize import minimize


from utilities.app_yfinance import tickers_yf, yf_ticket_info


@dataclass
class RollingStats:
    """ Rolling stats """
    rolling_mean: pd.DataFrame
    rolling_mean_z_score: pd.DataFrame
    rolling_std: pd.DataFrame
    rolling_std_z_score: pd.DataFrame
    rolling_sum: pd.DataFrame
    rolling_sum_z_score: pd.DataFrame
    rolling_return: pd.DataFrame
    rolling_return_z_score: pd.DataFrame
    rolling_annualised_return: pd.DataFrame
    rolling_annualised_return_z_score: pd.DataFrame


@dataclass
class AssetPosition:
    """ Asset position """
    symbol: str
    position: float


@dataclass
class PortfolioOptimizationResult:
    """ Portfolio optimization result """
    key: str
    name: str
    emoji: str
    weights: pd.Series
    log_return: float
    log_vol: float
    log_sharpe_ratio: float
    ann_log_return: float
    ann_vol: float
    ann_log_sharpe_ratio: float
    ann_arith_return: float
    ann_arith_sharpe_ratio: float
    herfindahl_index: float


@dataclass
class PeriodInfo:
    """ Period info """
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    number_of_points: int
    number_of_days: int
    points_per_day: float
    number_of_years: float
    points_per_year: float
    points_per_month: float


class Portfolio:
    """ Portfolio class """

    def __init__(self, asset_positions: list[AssetPosition], reference_currency: str):
        positions_series = pd.Series(
            data=[ass_pos.position for ass_pos in asset_positions],
            index=[ass_pos.symbol for ass_pos in asset_positions],
        )

        asset_symbols = [ass_pos.symbol for ass_pos in asset_positions]

        reference_currency = reference_currency.upper()
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.symbols_info_list = [yf_ticket_info(
            symbol) for symbol in asset_symbols]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        currencies_series = pd.Series(
            data=[info['currency']
                  for info in self.symbols_info_list]+[reference_currency],
            index=[ass_pos.symbol for ass_pos in asset_positions]+['TOTAL'],
        )

        currency_symbols = [(target_currency + reference_currency + "=X")
                            for target_currency in currencies_series.unique() if target_currency != reference_currency]

        # >>>
        self.tickers_data = tickers_yf(
            asset_symbols+currency_symbols, period='max')
        # <<<

        translated_close, translated_values = self._get_translated_history(
            asset_positions, reference_currency, currency_symbols)

        weights = translated_values.div(translated_values.sum(axis=1), axis=0)

        position_df = pd.DataFrame(
            data=[positions_series.reindex(
                translated_values.columns)] * len(translated_values),
            index=translated_values.index,
            columns=translated_values.columns)

        # Add a total column to the translated values and weights
        translated_values['TOTAL'] = translated_values.sum(axis=1)
        weights['TOTAL'] = weights.sum(axis=1)

        translated_fitted_values, trend_deviation, trend_deviation_z_score, cagr_fitted = self._get_fitted_values(
            translated_values)

        log_returns, cumulative_log_returns, annualized_to_date_return, \
            rolling_stats_1m, rolling_stats_1q, rolling_stats_1y, rolling_stats_3y = self._get_logarithmic_values(
                translated_values)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.timeseries_data = pd.concat(
            objs=[
                position_df,
                weights,
                translated_close,
                translated_values, translated_fitted_values,
                trend_deviation, trend_deviation_z_score,
                log_returns, cumulative_log_returns,
                annualized_to_date_return,
                rolling_stats_1m.rolling_return, rolling_stats_1q.rolling_return,
                rolling_stats_1y.rolling_return, rolling_stats_3y.rolling_return,
                rolling_stats_1m.rolling_return_z_score, rolling_stats_1q.rolling_return_z_score,
                rolling_stats_1y.rolling_return_z_score, rolling_stats_3y.rolling_return_z_score],
            axis=1,
            keys=[
                'position',
                'weights',
                'translated_close',
                'translated_values', 'translated_fitted_values',
                'trend_deviation', 'trend_deviation_z_score',
                'log_returns', 'cumulative_log_returns',
                'annualized_to_date_return',
                'rolling_1m_return', 'rolling_1q_return',
                'rolling_1y_return', 'rolling_3y_return',
                'rolling_1m_return_z_score', 'rolling_1q_return_z_score',
                'rolling_1y_return_z_score', 'rolling_3y_return_z_score'])

        self.timeseries_data.columns.names = ['Metric', 'Ticker']
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.optimisation_results = self._calculate_portfolio_optimisation(
            log_returns)

        self.assets_snapshot: pd.DataFrame = self.timeseries_data.iloc[-1].unstack(
            level='Metric')

        # Add additional metrics
        self.assets_snapshot['currency'] = currencies_series
        self.assets_snapshot['cagr_fitted'] = cagr_fitted
        self.assets_snapshot['cagr_fitted_pct'] = cagr_fitted.mul(100)
        self.assets_snapshot['latest_weights_pct'] = self.assets_snapshot['weights'].mul(
            100)
        self.assets_snapshot['cagr_pct'] = self.assets_snapshot['annualized_to_date_return'].mul(
            100)
        self.assets_snapshot['trend_deviation_pct'] = self.assets_snapshot['trend_deviation'].mul(
            100)
        self.assets_snapshot['rolling_1m_return_pct'] = self.assets_snapshot['rolling_1m_return'].mul(
            100)
        self.assets_snapshot['rolling_1q_return_pct'] = self.assets_snapshot['rolling_1q_return'].mul(
            100)
        self.assets_snapshot['rolling_1y_return_pct'] = self.assets_snapshot['rolling_1y_return'].mul(
            100)
        self.assets_snapshot['rolling_3y_return_pct'] = self.assets_snapshot['rolling_3y_return'].mul(
            100)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_assets_snapshot(self) -> pd.DataFrame:
        """ Get the assets snapshot """
        return self.assets_snapshot

    def get_optimisation_results(self) -> list[PortfolioOptimizationResult]:
        """ Get the optimization results """
        return self.optimisation_results

    def get_asset_series(self, p_asset: str):
        """ Get the series for the given asset """

        if not p_asset in self.timeseries_data.columns.get_level_values('Ticker').unique():
            raise ValueError(f"Asset {p_asset} not found in timeseries data")

        return self.timeseries_data.xs(p_asset, level='Ticker', axis=1)

    def get_period_info(self) -> PeriodInfo:
        """ Get the period info for the given history """

        first_date = self.tickers_data['history'].index[0]
        last_date = self.tickers_data['history'].index[-1]

        if not isinstance(first_date, pd.Timestamp):
            raise ValueError("First date is not a Timestamp")
        if not isinstance(last_date, pd.Timestamp):
            raise ValueError("Last date is not a Timestamp")

        number_of_points = self.tickers_data['history'].shape[0]
        number_of_days = (last_date - first_date).days + 1

        points_per_day = number_of_points / number_of_days
        number_of_years = number_of_days / 365.25
        points_per_year = number_of_points / number_of_years
        points_per_month = points_per_year / 12

        return PeriodInfo(
            first_date=first_date,
            last_date=last_date,
            number_of_points=number_of_points,
            number_of_days=number_of_days,
            points_per_day=points_per_day,
            number_of_years=number_of_years,
            points_per_year=points_per_year,
            points_per_month=points_per_month
        )

    def _get_logarithmic_values(self, translated_values: pd.DataFrame):
        """ Get logarithmic values for the given history """

        period_info = self.get_period_info()
        points_per_year = period_info.points_per_year

        log_returns = np.log(translated_values /
                             translated_values.shift(1)).dropna()

        if not isinstance(log_returns, pd.DataFrame):
            raise ValueError("Log returns is not a DataFrame")

        cumulative_log_returns = log_returns.cumsum()

        if not isinstance(cumulative_log_returns, pd.DataFrame):
            raise ValueError("Cumulative log returns is not a DataFrame")

        annualized_to_date_return = np.exp(
            cumulative_log_returns*period_info.points_per_year/cumulative_log_returns.shape[0]) - 1

        if not isinstance(annualized_to_date_return, pd.DataFrame):
            raise ValueError(
                "Annualized to date return is not a DataFrame")

        # just remove 1 year of data (to remove starting values which are created from very few data points)
        annualized_to_date_return = annualized_to_date_return.iloc[round(
            points_per_year * 1):]

        rolling_stats_1m = self._get_rolling_stats(
            log_returns, period_info.points_per_month)

        rolling_stats_1q = self._get_rolling_stats(
            log_returns, period_info.points_per_year / 4)

        rolling_stats_1y = self._get_rolling_stats(
            log_returns, period_info.points_per_year)

        rolling_stats_3y = self._get_rolling_stats(
            log_returns, period_info.points_per_year * 3)

        return log_returns, cumulative_log_returns, annualized_to_date_return, \
            rolling_stats_1m, rolling_stats_1q, rolling_stats_1y, rolling_stats_3y

    def _get_rolling_stats(self, p_log_returns: pd.DataFrame, p_window: float):
        """ Get rolling stats for the given log returns """
        period_info = self.get_period_info()

        window = round(p_window)

        rolling = p_log_returns.rolling(window=window)
        if not isinstance(rolling, Rolling):
            raise ValueError("Rolling is not a Rolling")

        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        rolling_sum = rolling.sum()

        rolling_return = np.exp(rolling_sum) - 1

        rolling_annualised_return = np.exp(
            rolling_sum * period_info.points_per_year / window) - 1

        if not isinstance(rolling_mean, pd.DataFrame):
            raise ValueError("Rolling mean is not a DataFrame")
        if not isinstance(rolling_std, pd.DataFrame):
            raise ValueError("Rolling std is not a DataFrame")
        if not isinstance(rolling_sum, pd.DataFrame):
            raise ValueError("Rolling sum is not a DataFrame")
        if not isinstance(rolling_return, pd.DataFrame):
            raise ValueError("Rolling return is not a DataFrame")

        rolling_mean_z_score = (
            rolling_mean - rolling_mean.mean()) / rolling_mean.std()

        rolling_std_z_score = (
            rolling_std - rolling_std.mean()) / rolling_std.std()

        rolling_sum_z_score = (
            rolling_sum - rolling_sum.mean()) / rolling_sum.std()

        rolling_return_z_score = (
            rolling_return - rolling_return.mean()) / rolling_return.std()

        rolling_annualised_return_z_score = (
            rolling_annualised_return - rolling_annualised_return.mean()) / rolling_annualised_return.std()

        return RollingStats(
            rolling_mean=rolling_mean,
            rolling_mean_z_score=rolling_mean_z_score,
            rolling_std=rolling_std,
            rolling_std_z_score=rolling_std_z_score,
            rolling_sum=rolling_sum,
            rolling_sum_z_score=rolling_sum_z_score,
            rolling_return=rolling_return,
            rolling_return_z_score=rolling_return_z_score,
            rolling_annualised_return=rolling_annualised_return,
            rolling_annualised_return_z_score=rolling_annualised_return_z_score)

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

        cagr_fitted = (fitted_values.iloc[-1] / fitted_values.iloc[0]) ** (
            1 / self.get_period_info().number_of_years) - 1

        if not isinstance(cagr_fitted, pd.Series):
            raise ValueError("CAGR fitted is not a Series")

        return fitted_values, trend_deviation, trend_deviation_z_score, cagr_fitted

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
        translated_close = assets_history.copy()

        # Create a dictionary to map asset_symbols to positions for quick lookup
        positions_dict = {ass_pos.symbol: ass_pos.position
                          for ass_pos in asset_positions}

        for symbol, info in zip(assets_history.columns, self.symbols_info_list):
            position_value = positions_dict[symbol]

            translated_values[symbol] = translated_values[symbol] * \
                position_value

            if info['currency'] != reference_currency:
                currency_column = info['currency'] + reference_currency + "=X"
                translated_values[symbol] = translated_values[symbol] * \
                    currency_history[currency_column]

                translated_close[symbol] = translated_close[symbol] * \
                    currency_history[currency_column]

        return translated_close, translated_values

    def _calculate_portfolio_optimisation(self, p_log_returns: pd.DataFrame):
        """ Calculate optimal weights  """
        risk_free_rate = 0

        # Minimum Volatility Portfolio
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Maximum Sharpe Ratio Portfolio
        def neg_sharpe_ratio(weights, expected_log_returns, cov_log_matrix, risk_free_rate):

            port_log_return = np.sum(weights * expected_log_returns)
            port_log_vol = portfolio_volatility(weights, cov_log_matrix)

            if port_log_vol > 0:
                return -(port_log_return - risk_free_rate) / port_log_vol
            else:
                return 0

        # Risk Parity Portfolio
        def risk_parity_objective(weights, cov_matrix):

            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

            marginal_contrib = np.dot(cov_matrix, weights)

            risk_contrib = weights * marginal_contrib / portfolio_variance

            target_contrib = np.ones(len(weights)) / len(weights)

            return np.sum((risk_contrib - target_contrib) ** 2)

        # Get asset data (excluding TOTAL column and currency conversion symbols)
        log_returns = p_log_returns.copy()
        log_returns = log_returns.drop(columns=['TOTAL'], errors='ignore')
        mean_log_returns = log_returns.mean()
        if not isinstance(mean_log_returns, pd.Series):
            raise ValueError("Mean log returns is not a Series")

        cov_log_matrix = log_returns.cov()
        if not isinstance(cov_log_matrix, pd.DataFrame):
            raise ValueError("Covariance matrix is not a DataFrame")

        # Use points_per_year for annualization
        period_info = self.get_period_info()
        points_per_year = period_info.points_per_year

        # Common optimization setup
        n_assets = len(log_returns.columns)
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Helper function to calculate portfolio metrics
        def calculate_portfolio_metrics(weights, key, name, emoji):
            port_log_return = np.sum(weights * mean_log_returns)
            port_log_vol = np.sqrt(
                np.dot(weights.T, np.dot(cov_log_matrix, weights)))
            port_log_sharpe_ratio = (
                port_log_return - risk_free_rate) / port_log_vol if port_log_vol > 0 else 0

            ann_log_return = port_log_return * points_per_year
            ann_vol = port_log_vol * np.sqrt(points_per_year)
            ann_log_sharpe_ratio = (
                ann_log_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

            ann_arith_return = np.exp(ann_log_return) - 1
            ann_arith_sharpe_ratio = (
                ann_arith_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

            herfindahl_index = np.sum(weights ** 2)

            return PortfolioOptimizationResult(
                key=key,
                name=name,
                emoji=emoji,
                weights=weights,
                log_return=port_log_return,
                log_vol=port_log_vol,
                log_sharpe_ratio=port_log_sharpe_ratio,
                ann_log_return=ann_log_return,
                ann_vol=ann_vol,
                ann_log_sharpe_ratio=ann_log_sharpe_ratio,
                ann_arith_return=ann_arith_return,
                ann_arith_sharpe_ratio=ann_arith_sharpe_ratio,
                herfindahl_index=herfindahl_index
            )

        # ================================
        # PORTFOLIO OPTIMIZATION - MULTIPLE STRATEGIES
        # ================================

        # Initialize with current portfolio
        optimal_weights: list[PortfolioOptimizationResult] = [calculate_portfolio_metrics(
            self.timeseries_data.xs(
                'weights', level='Metric', axis=1).iloc[-1].drop('TOTAL', errors='ignore'),
            'current_portfolio',
            'Current Portfolio',
            '💼')]

        # Define optimization strategies
        strategies = [
            {
                'name': 'Maximum Sharpe Ratio',
                'emoji': '🎯',
                'key': 'max_sharpe',
                'objective_func': neg_sharpe_ratio,
                'args': (mean_log_returns, cov_log_matrix, risk_free_rate),
                'initial_guess': np.array([1.0/n_assets] * n_assets)
            },
            {
                'name': 'Risk Parity',
                'emoji': '⚖️',
                'key': 'risk_parity',
                'objective_func': risk_parity_objective,
                'args': (cov_log_matrix),
                'initial_guess': np.array([1.0/n_assets] * n_assets)
            },
            {
                'name': 'Minimum Volatility',
                'emoji': '🛡️',
                'key': 'min_volatility',
                'objective_func': portfolio_volatility,
                'args': (cov_log_matrix),
                'initial_guess': (1 / np.diag(cov_log_matrix.values)) / np.sum(1 / np.diag(cov_log_matrix.values))
            }
        ]

        # Loop through each strategy
        for strategy in strategies:

            result = minimize(strategy['objective_func'], strategy['initial_guess'],
                              args=strategy['args'],
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = pd.Series(
                    result.x, index=mean_log_returns.index)
                optimal_weights.append(calculate_portfolio_metrics(
                    weights, strategy['key'], strategy['name'], strategy['emoji']))

            else:
                print(
                    f"❌ {strategy['name']} optimization failed: {result.message}")

        # Sort by ann_vol ascending
        optimal_weights.sort(key=lambda x: x.ann_vol)

        # Validations - exclude current portfolio from min volatility validation
        min_volatility_results = [
            r for r in optimal_weights if r.key == 'min_volatility']
        if min_volatility_results:
            min_volatility_result = min_volatility_results[0]
            other_results = [
                r for r in optimal_weights
                if r.key not in ['min_volatility', 'current_portfolio']
            ]

            for other_result in other_results:
                if min_volatility_result.ann_vol > other_result.ann_vol:
                    raise ValueError(f"⚠️ Warning: Min Volatility strategy ({min_volatility_result.ann_vol:.4f}) "
                                     f"has higher volatility than {other_result.name} ({other_result.ann_vol:.4f})")

        return optimal_weights
