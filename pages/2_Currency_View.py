
import datetime
import streamlit as st
from utilities.constants import BASE_CURRENCY_OPTIONS
from utilities.utilities import get_fx_history_2


print(f"\n--- Currency view {datetime.datetime.now()} ---\n")
st.title("Currency and Cyptocurrency Information")

col1, col2 = st.columns(2)

with col1:
    base_currency = st.selectbox("Select base currency",
                                 BASE_CURRENCY_OPTIONS, key="base_currency_input")

with col2:
    target_currency = st.text_input(
        "Select target currency or crypto",  key="target_currency_input")

if target_currency:

    history, symbol = get_fx_history_2(base_currency, target_currency)

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
