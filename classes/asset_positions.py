from dataclasses import dataclass
import numpy as np
import pandas as pd

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

    def __init__(self, positions: list[AssetPosition], reference_currency: str):
        self.positions = positions
        self.reference_currency = reference_currency.upper()

        asset_symbols = [position.get_symbol() for position in self.positions]
        self.symbols_info_list = [yf_ticket_info(
            symbol) for symbol in asset_symbols]
        unique_currencies = set([info['currency']
                                 for info in self.symbols_info_list])

        self.currency_symbols = [(target_currency + self.reference_currency + "=X")
                                 for target_currency in unique_currencies if target_currency != self.reference_currency]

        self.history_combo = tickers_yf(
            asset_symbols+self.currency_symbols, period='max')

        print(f"self.history_combo: \n{self.history_combo}")

        translated_values = self._get_translated_history()

        weights = translated_values.div(translated_values.sum(axis=1), axis=0)

        translated_values['TOTAL'] = translated_values.sum(axis=1)

        translated_fitted_values = self._get_fitted_values(translated_values)

        trend_deviation = (translated_values / translated_fitted_values) - 1
        trend_deviation_z_score = (trend_deviation - trend_deviation.mean()) / \
            trend_deviation.std()

        first_date, last_date, number_of_points, number_of_days, number_of_years, points_per_year, points_per_month = self.get_period_info()

        daily_log_returns: pd.DataFrame = np.log(
            translated_values / translated_values.shift(1))
        monthly_log_returns: pd.DataFrame = np.log(
            translated_values / translated_values.shift(round(points_per_month)))
        yearly_log_returns: pd.DataFrame = np.log(
            translated_values / translated_values.shift(round(points_per_year)))

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.timeseries_data = pd.concat(
            objs=[
                translated_values, translated_fitted_values, trend_deviation, trend_deviation_z_score,
                daily_log_returns, monthly_log_returns, yearly_log_returns],
            axis=1,
            keys=['translated_values', 'translated_fitted_values', 'trend_deviation', 'trend_deviation_z_score',
                  'daily_log_returns', 'monthly_log_returns', 'yearly_log_returns'])

        self.timeseries_data.columns.names = ['Metric', 'Ticker']
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        first_value, last_value, first_fitted_value, last_fitted_value = self.get_first_last_values(
            translated_values, translated_fitted_values)

        cagr: pd.Series = (
            last_value / first_value) ** (1 / number_of_years) - 1
        cagr_fitted: pd.Series = (last_fitted_value /
                                  first_fitted_value) ** (1 / number_of_years) - 1

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.assets_metrics = pd.concat(
            objs=[
                cagr, cagr_fitted,
                cagr.mul(100), cagr_fitted.mul(100),
                pd.Series(index=cagr.index, data=first_date), pd.Series(
                    index=cagr.index, data=last_date),
                last_value, last_fitted_value],
            axis=1,
            keys=[
                'cagr', 'cagr_fitted',
                'cagr_pct', 'cagr_fitted_pct',
                'first_date', 'last_date',
                'last_value', 'last_fitted_value'])
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def get_assets_metrics(self):
        """ Get the assets metrics """
        return self.assets_metrics

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
        first_date = self.history_combo['history'].index[0]
        last_date = self.history_combo['history'].index[-1]

        if not isinstance(first_date, pd.Timestamp):
            raise ValueError("First date is not a Timestamp")
        if not isinstance(last_date, pd.Timestamp):
            raise ValueError("Last date is not a Timestamp")

        number_of_points = self.history_combo['history'].shape[0]
        number_of_days = (last_date - first_date).days + 1
        number_of_years = number_of_days / 365.25
        points_per_year = number_of_points / number_of_years
        points_per_month = points_per_year / 12

        return first_date, last_date, number_of_points, number_of_days, number_of_years, points_per_year, points_per_month

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

        return fitted_values

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

    def _get_translated_history(self):

        assets_history = self.history_combo['history'].drop(
            columns=self.currency_symbols)
        currency_history = self.history_combo['history'][self.currency_symbols]

        if not isinstance(currency_history, pd.DataFrame):
            raise ValueError("Currency history is not a DataFrame")

        translated_values = assets_history.copy()

        # Create a dictionary to map asset_symbols to positions for quick lookup
        positions_dict = {pos.get_symbol(): pos.get_position()
                          for pos in self.positions}

        for symbol, info in zip(assets_history.columns, self.symbols_info_list):
            position_value = positions_dict[symbol]

            translated_values[symbol] = translated_values[symbol] * \
                position_value

            if info['currency'] != self.reference_currency:
                currency_column = info['currency'] + \
                    self.reference_currency + "=X"
                translated_values[symbol] = translated_values[symbol] * \
                    currency_history[currency_column]

        return translated_values
