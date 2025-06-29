from dataclasses import dataclass


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
        info = [yf_ticket_info(symbol) for symbol in symbols]
        unique_currencies = set([info['currency'] for info in info])

        currency_symbols = [(target_currency + self.reference_currency + "=X")
                            for target_currency in unique_currencies if target_currency != self.reference_currency]

        history_all = tickers_yf(symbols+currency_symbols, period='max')

        symbols_history = history_all['history'].drop(columns=currency_symbols)
        currency_history = history_all['history'][currency_symbols]

        print(f"symbols_history: {symbols_history}")
        print(f"currency_history: {currency_history}")
        # print(info)

        # Create translated DataFrame
        # self.translated_values = self._create_translated_values(
        #     history['history'], info)

        # print(self.translated_values)

    def get_positions(self) -> list[AssetPosition]:
        """ Get the positions """
        return self.positions
