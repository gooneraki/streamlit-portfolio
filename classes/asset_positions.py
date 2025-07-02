""" Module for asset positions and portfolio """

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


from utilities.app_yfinance import tickers_yf, yf_ticket_info


class AssetPosition:
    """ Asset position """

    def __init__(self, symbol: str, position: float):
        self.symbol = symbol
        self.position = position

    def get_symbol(self) -> str:
        """ Get the symbol """
        return self.symbol

    def get_position(self) -> float:
        """ Get the position """
        return self.position


@dataclass
class PortfolioOptimizationResult:
    """ Portfolio optimization result """
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
    years_series: pd.Series


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

        log_returns, cumulative_log_returns, cumulative_arithmetic_returns = self._get_logarithmic_values(
            translated_values)

        self.optimal_weights = self._calculate_portfolio_optimization(
            log_returns)

        period_info = self.get_period_info()
        number_of_years = period_info.number_of_years

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.timeseries_data = pd.concat(
            objs=[
                weights,
                translated_values, translated_fitted_values,
                trend_deviation, trend_deviation_z_score,
                log_returns, cumulative_log_returns,
                cumulative_arithmetic_returns],
            axis=1,
            keys=[
                'weights',
                'translated_values', 'translated_fitted_values',
                'trend_deviation', 'trend_deviation_z_score',
                'log_returns', 'cumulative_log_returns',
                'cumulative_arithmetic_returns'])

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

    def get_assets_metrics(self) -> pd.DataFrame:
        """ Get the assets metrics """
        return self.assets_metrics

    def get_optimal_weights(self) -> dict[str, PortfolioOptimizationResult]:
        """ Get the optimal weights """
        return self.optimal_weights

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

    def get_period_info(self) -> PeriodInfo:
        """ Get the period info for the given history """
        period_series = pd.Series(
            data=np.arange(0, len(self.tickers_data['history'])),
            index=self.tickers_data['history'].index,
            name='period'
        )

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

        years_series = period_series / points_per_year
        if not isinstance(years_series, pd.Series):
            raise ValueError("Years series is not a Series")

        return PeriodInfo(
            first_date=first_date,
            last_date=last_date,
            number_of_points=number_of_points,
            number_of_days=number_of_days,
            points_per_day=points_per_day,
            number_of_years=number_of_years,
            points_per_year=points_per_year,
            points_per_month=points_per_month,
            years_series=years_series
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

        # Annualize using correct exponent for each row
        cumulative_arithmetic_returns = (pd.DataFrame(
            np.exp(cumulative_log_returns),
            index=cumulative_log_returns.index,
            columns=cumulative_log_returns.columns
        ).pow(1/period_info.years_series, axis=0) - 1).dropna()

        if not isinstance(cumulative_arithmetic_returns, pd.DataFrame):
            raise ValueError(
                "Cumulative arithmetic returns is not a DataFrame")

        # just remove 1 year of data (to remove starting values which are created from very few data points)
        cumulative_arithmetic_returns = cumulative_arithmetic_returns.iloc[round(
            points_per_year * 1):]

        return log_returns, cumulative_log_returns, cumulative_arithmetic_returns

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

    def _calculate_portfolio_optimization(self, p_log_returns: pd.DataFrame):
        """ Calculate optimal weights  """
        risk_free_rate = 0

        # Maximum Return Portfolio
        def neg_portfolio_return(weights, expected_log_returns):
            return -np.sum(weights * expected_log_returns)

        # Minimum Volatility Portfolio
        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Maximum Sharpe Ratio Portfolio
        def neg_sharpe_ratio(weights, expected_log_returns, cov_log_matrix, risk_free_rate):

            neg_port_log_return = neg_portfolio_return(
                weights, expected_log_returns)
            port_log_vol = portfolio_volatility(weights, cov_log_matrix)

            if port_log_vol > 0:
                return (neg_port_log_return - risk_free_rate) / port_log_vol
            else:
                return 0

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
        initial_guess = np.array([1.0/n_assets] * n_assets)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # ================================
        # PORTFOLIO OPTIMIZATION - MULTIPLE STRATEGIES
        # ================================

        optimal_weights_dict: dict[str, PortfolioOptimizationResult] = {}

        # Define optimization strategies
        strategies = [
            {
                'name': 'Maximum Sharpe Ratio',
                'emoji': '🎯',
                'key': 'max_sharpe',
                'objective_func': neg_sharpe_ratio,
                'args': (mean_log_returns, cov_log_matrix, risk_free_rate)
            },
            {
                'name': 'Maximum Return',
                'emoji': '🚀',
                'key': 'max_return',
                'objective_func': neg_portfolio_return,
                'args': (mean_log_returns,)
            },
            {
                'name': 'Minimum Volatility',
                'emoji': '🛡️',
                'key': 'min_volatility',
                'objective_func': portfolio_volatility,
                'args': (cov_log_matrix,)
            }
        ]

        # Loop through each strategy
        for strategy in strategies:

            result = minimize(strategy['objective_func'], initial_guess,
                              args=strategy['args'],
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = pd.Series(
                    result.x, index=mean_log_returns.index)

                # Daily logarithmic return and volatility
                port_log_return = np.sum(optimal_weights * mean_log_returns)
                port_log_vol = np.sqrt(
                    np.dot(optimal_weights.T, np.dot(cov_log_matrix, optimal_weights)))
                port_log_sharpe_ratio = (
                    port_log_return - risk_free_rate) / port_log_vol

                # Annualized logarithmic return and volatility
                ann_log_return = port_log_return * points_per_year
                ann_vol = port_log_vol * np.sqrt(points_per_year)
                ann_log_sharpe_ratio = (
                    ann_log_return - risk_free_rate) / ann_vol

                # Annualized arithmetic return and sharpe ratio
                ann_arith_return = np.exp(ann_log_return) - 1
                ann_arith_sharpe_ratio = (ann_arith_return - risk_free_rate) / \
                    ann_vol if ann_vol > 0 else 0

                optimal_weights_dict[strategy['key']] = PortfolioOptimizationResult(
                    name=strategy['name'],
                    emoji=strategy['emoji'],
                    weights=optimal_weights,

                    log_return=port_log_return,
                    log_vol=port_log_vol,
                    log_sharpe_ratio=port_log_sharpe_ratio,

                    ann_log_return=ann_log_return,
                    ann_vol=ann_vol,
                    ann_log_sharpe_ratio=ann_log_sharpe_ratio,
                    ann_arith_return=ann_arith_return,
                    ann_arith_sharpe_ratio=ann_arith_sharpe_ratio,
                )

            else:
                print(
                    f"❌ {strategy['name']} optimization failed. {result.message}")

        return optimal_weights_dict
