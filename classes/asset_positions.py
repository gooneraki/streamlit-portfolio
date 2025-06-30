from dataclasses import dataclass
import numpy as np
import pandas as pd

from utilities.app_yfinance import tickers_yf, yf_ticket_info


@dataclass
class AssetPosition:
    """ Asset position """
    symbol: str
    position: float


class Portfolio:

    def __init__(self, positions: list[AssetPosition], reference_currency: str):
        self.positions = positions
        self.reference_currency = reference_currency.upper()

        symbols = [position['symbol'] for position in self.positions]
        symbols_info_list = [yf_ticket_info(symbol) for symbol in symbols]
        unique_currencies = set([info['currency']
                                for info in symbols_info_list])

        currency_symbols = [(target_currency + self.reference_currency + "=X")
                            for target_currency in unique_currencies if target_currency != self.reference_currency]

        history_all = tickers_yf(symbols+currency_symbols, period='max')

        symbols_history = history_all['history'].drop(columns=currency_symbols)
        currency_history = history_all['history'][currency_symbols]

        translated_values = symbols_history if currency_history.empty else self._create_translated_values(
            symbols_history,
            currency_history,
            symbols_info_list)

        translated_fitted_values = self._get_fitted_values(translated_values)

        translated_total = translated_values.sum(axis=1)

        translated_fitted_total = self._get_fitted_total_values(
            translated_total)

    def get_positions(self) -> list[AssetPosition]:
        """ Get the positions """
        return self.positions

    def _create_translated_values(
            self,
            symbols_history: pd.DataFrame,
            currency_history: pd.DataFrame,
            symbols_info_list: list[dict]):
        """ Create translated values """
        translated_values = symbols_history.copy()

        for symbol, info in zip(symbols_history.columns, symbols_info_list):
            if info['currency'] != self.reference_currency:
                currency_column = info['currency'] + \
                    self.reference_currency + "=X"
                translated_values[symbol] = translated_values[symbol] * \
                    currency_history[currency_column]

        return translated_values

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
