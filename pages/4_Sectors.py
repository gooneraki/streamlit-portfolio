""" Streamlit app to display sector and industry information """
# pylint: disable=C0103
import datetime
import streamlit as st
import pandas as pd
from utilities.app_yfinance import YF_SECTOR_KEYS, sector_yf, market_yf, yf_ticket_info
from utilities.utilities import fully_analyze_symbol
from utilities.go_charts import display_trend_go_chart


print(f"\n--- Sectors view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="Sector Market Share", layout="centered")


market = market_yf('US')
if market is None:
    st.error("Error retrieving US market data.")
    st.stop()


market_symbol = market['summary'][list(market['summary'].keys())[
    0]].get('symbol', 'UNDEFINED')

market_info = yf_ticket_info(market_symbol)
analysis = fully_analyze_symbol(market_symbol, 'EUR', 10)
# reset index and set it as Date
analysis = analysis.reset_index()
analysis['Date'] = pd.to_datetime(analysis['Date'], errors='coerce')
value_fig = display_trend_go_chart(
    analysis,  title_name="US Market Value - 10 years",)

st.title("Sector Market Share in US")


if value_fig is None:
    st.warning("No valid data to plot.")
else:
    st.plotly_chart(value_fig, config={
                    "displayModeBar": False}, use_container_width=True)


st.write("This page displays the market share of all sectors in the US.")


sector_data = sorted([sector_yf(sectorInfo) for sectorInfo in YF_SECTOR_KEYS],
                     key=lambda x: x.get('overview', {}).get("market_weight", -9.99), reverse=True)

st.write("### Sectors Overview")
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

    with st.expander(sector_name, expanded=True):

        st.write(
            f"### {sector_name}")

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
            # convert dict to pandas
            top_etfs_df = pd.DataFrame.from_dict(
                top_etfs, orient='index', columns=['ETF']).reset_index(names='Symbol')
            st.write("**Top ETFs**")
            st.dataframe(top_etfs_df, hide_index=True)
