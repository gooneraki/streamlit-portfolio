from classes.asset_positions import AssetPosition


BASE_CURRENCY_OPTIONS = ["EUR", "USD",
                         "GBP", "JPY", "AUD", "CAD", "CHF", "HKD"]

ASSETS_POSITIONS_DEFAULT: list[AssetPosition] = [
    AssetPosition(symbol="CSPX.L", position=1),
    AssetPosition(symbol="AAPL", position=1),
    AssetPosition(symbol="MSFT", position=1),
    AssetPosition(symbol="GOOGL", position=1),
    AssetPosition(symbol="AMZN", position=1),
    AssetPosition(symbol="NVDA", position=1),
    AssetPosition(symbol="TSLA", position=1),
    AssetPosition(symbol="META", position=1),
    AssetPosition(symbol="BRK-B", position=1),
    AssetPosition(symbol="JPM", position=1),
]
