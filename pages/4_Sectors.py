"""Streamlit app to display sector and industry information"""

# pylint: disable=C0103
import datetime
import streamlit as st
import pandas as pd
from utilities.app_yfinance import YF_SECTOR_KEYS, sector_yf, market_yf, yf_ticket_info
from utilities.utilities import fully_analyze_symbol, metrics, retrieve_sector_data
from utilities.go_charts import display_trend_go_chart_2


print(f"\n--- Sectors view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="US Market Overview", layout="centered")

# User contants
home_currency = "EUR"
data_years = 10
# country = "US"


# Fetch market data
# market = market_yf(country)
# if market is None:
#     st.error("Error retrieving US market data.")
#     st.stop()

# # print market keys
# print(f"Market keys: {list(market.keys())}")

# Get first market symbol
# market_symbol = market['summary'][list(market['summary'].keys())[
#     0]].get('symbol', 'UNDEFINED')

market_symbol = "^GSPC"

market_info = yf_ticket_info(market_symbol)
trade_df, trade_metrics, home_df, home_metrics = fully_analyze_symbol(
    market_symbol, home_currency, data_years
)


# Fetch sector data
sector_data, sector_data_df = retrieve_sector_data(home_currency, data_years)


# UI
st.title("US Market Overview")

st.write(
    f"Home currency: **{home_currency}** || Data window: **{data_years} years**")

#
# MARKET
#
st.write("### Market as a whole")


st.write(
    f"Name: **{market_info['shortName']}** || " +
    f"Trade Currency: **{market_info['currency']}** || " +
    f"Symbol: **{market_symbol}**")

# Market Chart
col1, col2, _ = st.columns([1, 1, 2])
with col1:
    years_to_show = st.selectbox(
        "Years displayed",
        options=[1, 2, 3, 5, 7, 10],
    )

currency_options = ['Trade (USD)', 'Home (' + home_currency + ')']
selected_currency = currency_options[1]

with col2:
    selected_currency = st.selectbox(
        "Currency",
        options=currency_options,
        index=1
    )

filtered_first_date = trade_metrics['last_date'] - pd.DateOffset(
    years=years_to_show) if selected_currency == currency_options[0] else home_metrics['last_date'] - pd.DateOffset(years=years_to_show)

value_fig = display_trend_go_chart_2(
    trade_df[trade_df.index >= filtered_first_date] if selected_currency == currency_options[0] else home_df[home_df.index >= filtered_first_date],
    'Trade Value' if selected_currency == currency_options[0] else 'Home Value',
    'Trade Value Fitted' if selected_currency == currency_options[0] else 'Home Value Fitted',
    title_name=f"{market_info['shortName']}")

st.plotly_chart(
    value_fig,
    config={
        'staticPlot': True},
    use_container_width=True)

metrics_df = pd.DataFrame({
    "Trade Currency": trade_metrics,
    "Home Currency": home_metrics
})
metrics_df.drop([metrics['start_date'].key,
                 metrics['actual_years_duration'].key,
                metrics['last_date'].key], inplace=True)

metrics_df.index = metrics_df.index.map(lambda x: metrics[x].label)

st.dataframe(metrics_df.style.format(
    formatter=lambda x: f"{x:.1%}" if isinstance(x, float) else x
))


st.write("### Sectors")
st.dataframe(sector_data_df.style.format(
    formatter=lambda x: f"{x:.1%}" if isinstance(x, float) else x

), hide_index=True)


for faulty_sector in [
        sector for sector in sector_data if isinstance(sector, str)]:
    st.warning(faulty_sector)


for sector in sector_data:
    sector_name = sector.get('name', 'UNDEFINED')
    overview = sector.get('overview')
    top_companies = sector.get('top_companies')
    top_etfs = sector.get('top_etfs')

    with st.expander(sector_name, expanded=False):

        st.write(
            f"#### {sector_name}")

        if overview is None:
            st.warning(f"No sector overview found. '{sector_name}'")
        else:
            st.write(
                f'**Market Weight**: {100*overview.get('market_weight', -9.99):.1f}%')

            st.write(
                f'**Description**: {overview.get("description", "No description available.")}')

        if top_companies is None:
            st.warning(f"No top companies found. '{sector_name}'")
        else:
            st.write("**Top Companies**")
            st.dataframe(top_companies)

        if top_etfs is None:
            st.warning(f"No top ETFs found. '{sector_name}'")
        else:
            top_etfs_df = pd.DataFrame.from_dict(
                top_etfs, orient='index', columns=['ETF']).reset_index(names='Symbol')
            st.write("**Top ETFs**")
            st.dataframe(top_etfs_df, hide_index=True)
