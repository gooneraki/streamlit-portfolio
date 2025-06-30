from classes.asset_positions import AssetPosition


BASE_CURRENCY_OPTIONS = ["EUR", "USD",
                         "GBP", "JPY", "AUD", "CAD", "CHF", "HKD"]

ASSETS_POSITIONS_DEFAULT: list[AssetPosition] = [
    AssetPosition(symbol="CSPX.L", position=1),
    AssetPosition(symbol="IUSE.L", position=1),
    AssetPosition(symbol="IWDE.L", position=1),
    AssetPosition(symbol="IWDA.L", position=1)
]
