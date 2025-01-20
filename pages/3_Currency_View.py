import yfinance as yf
import datetime
import streamlit as st


@st.cache_data
def get_fx_history(base_currency, target_currency):
    """Get the historical data of a currency pair from Yahoo Finance API."""
    currency_symbol = target_currency + base_currency + "=X"
    crypto_symbol = target_currency + "-" + base_currency

    currency_rate_history = yf.Ticker(
        currency_symbol).history(period='max')['Close']
    crypto_rate_history = yf.Ticker(
        crypto_symbol).history(period='max')['Close']

    # final_fx_rate_history = None
    # final_symbol = None
    # if currency_rate_history.empty and crypto_rate_history.empty:
    #     return final_fx_rate_history, final_symbol
    # elif currency_rate_history.empty:
    #     return crypto_rate_history, crypto_symbol
    # else:
    #     return currency_rate_history, currency_symbol

    # return final_fx_rate_history, final_symbol
    return currency_rate_history if not currency_rate_history.empty else crypto_rate_history, currency_symbol if not currency_rate_history.empty else crypto_symbol


print(f"\n--- Currency view {datetime.datetime.now()} ---\n")
st.title("Currency and Cyptocurrency Information")

col1, col2 = st.columns(2)

with col1:
    base_currency = st.selectbox("Select base currency", [
        "EUR", "USD", "GBP", "JPY", "AUD", "CAD", "CHF", "HKD"], key="base_currency_input")

with col2:
    target_currency = st.text_input(
        "Select target currency or crypto",  key="target_currency_input")

if target_currency:

    history, symbol = get_fx_history(base_currency, target_currency)

    if history.empty:
        st.error("Currency not found.")
    else:
        symbol_type = "Currency" if '=X' in symbol else "Crypto"
        st.write(f"{symbol_type}: {symbol}")

        st.line_chart(history)

    # currency_symbol = target_currency + base_currency + "=X"
    # crypto_symbol = target_currency + "-" + base_currency

    # currency_rate_history = yf.Ticker(
    #     currency_symbol).history(period='max')['Close']
    # crypto_rate_history = yf.Ticker(
    #     crypto_symbol).history(period='max')['Close']

    # final_fx_rate_history = None
    # if currency_rate_history.empty and crypto_rate_history.empty:
    #     st.error("Currency not found.")
    # elif currency_rate_history.empty:
    #     st.write('Crypto')
    #     final_fx_rate_history = crypto_rate_history

    # else:
    #     st.write('Currency')
    #     final_fx_rate_history = currency_rate_history

    # if final_fx_rate_history is not None:

    #     st.write(final_fx_rate_history)

    #     # lets plot a trendlione
    #     st.line_chart(final_fx_rate_history,)
