""" Streamlit app to display sector and industry information """
# pylint: disable=C0103
import datetime
import streamlit as st
import pandas as pd
from utilities.app_yfinance import YF_SECTOR_KEYS, sector_yf, market_yf, yf_ticket_info
from utilities.utilities import fully_analyze_symbol, metrics
from utilities.go_charts import display_trend_go_chart_2


print(f"\n--- Sectors view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="US Market Overview", layout="centered")

# User contants
home_currency = "EUR"
data_years = 10
country = "US"

# Fetch market data
market = market_yf(country)
if market is None:
    st.error("Error retrieving US market data.")
    st.stop()

# # print market keys
# print(f"Market keys: {list(market.keys())}")


# Get first market symbol
market_symbol = market['summary'][list(market['summary'].keys())[
    0]].get('symbol', 'UNDEFINED')
# Fetch market info of the first symbol
market_info = yf_ticket_info(market_symbol)
# Fetch market history of the first symbol and stats
trade_df, trade_metrics, home_df, home_metrics = fully_analyze_symbol(
    market_symbol, home_currency, data_years)

# Fetch sector data
sector_data = sorted([sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS],
                     key=lambda x: x.get('overview', {}).get("market_weight", -9.99), reverse=True)


# Adjust for Chart
# market_history = market_analysis.get('ticker_history').reset_index()
# market_history['Date'] = pd.to_datetime(
#     market_history['Date'], errors='coerce')


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
# metrics_df.index = metrics_df.index.apply(
#     lambda x: metrics[x].label if x in metrics else x)
metrics_df.index = metrics_df.index.map(lambda x: metrics[x].label)
# metrics_df = metrics_df.T

st.dataframe(metrics_df)

# print(summary_data)
# print(f"Summary data: {summary_data}")


# st.dataframe(pd.DataFrame(
#     data={
#         "Trade Currency": trade_metrics,
#         "Home Currency": home_metrics
#     }))

# st.dataframe(pd.DataFrame(
#     trade_metrics if selected_currency ==
#              currency_options[0] else home_metrics))

# Market Table
# st.write(pd.DataFrame(
#     data=[
#         [
#             market_analysis.get('trade_currency_cagr'),
#             market_analysis.get('trade_cur_fitted_cagr'),
#             market_analysis.get('trade_cur_over_under'),
#             market_analysis.get('trade_cur_annual_returns_variance')],
#         [market_analysis.get('home_currency_cagr'),
#          market_analysis.get('home_cur_fitted_cagr'),
#          market_analysis.get('home_cur_over_under'),
#          market_analysis.get('home_cur_annual_returns_variance')]],
#     columns=[
#         'Annual Growth Rate',
#         'Fitted Annual Growth Rate',
#         'Over/Under valued as at ' +
#         pd.to_datetime(market_history.tail(
#             1)['Date'].values[0]).strftime('%d/%m/%y'),
#         'Annual Returns Variance'],
#     index=[f'Trade cur ({market_info.get('currency')})',
#            f'Home cur ({home_currency})']
# ).T.style.format(formatter=lambda x: f"{x:.1%}"))


st.write("### Sectors")
st.dataframe(pd.DataFrame(
    data=[[sector.get('name', 'UNDEFINED'), sector.get(
        'overview', {}).get("market_weight", -9.99)] for sector in sector_data],
    columns=['Sector', 'Market Weight']).style.format(
    {'Market Weight': '{:.1%}'}), hide_index=True)


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
