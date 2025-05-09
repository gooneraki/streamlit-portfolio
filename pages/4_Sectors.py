# import datetime
import streamlit as st
import pandas as pd
from utilities.app_yfinance import retrieve_sector_industry_keys, get_sector_details

# print(f"\n--- Sectors view: {datetime.datetime.now()} ---\n")

st.set_page_config(page_title="Sector Market Share", layout="centered")

sectors = retrieve_sector_industry_keys()
st.title("Sector Market Share in US")

st.write("This page displays the market share of all sectors in the US.")


sector_data = sorted([get_sector_details(sectorInfo) for sectorInfo in sectors],
                     key=lambda x: x.get('overview', {}).get("market_weight", 0), reverse=True)

st.write("### Sector Overview")
columns = ['Sector', 'Market Weight']
data = [[sector.get('sector_name').replace('-', ' ').title(), sector.get(
    'overview', {}).get("market_weight", -9.99)] for sector in sector_data]
df = pd.DataFrame(data, columns=columns)

st.dataframe(df.style.format(
    {'Market Weight': '{:.1%}'}), hide_index=True)

for sector in sector_data:
    sector_key = sector.get('sector_name')
    overview = sector.get('overview')
    top_companies = sector.get('top_companies')
    top_etfs = sector.get('top_etfs')

    st.write(
        f"### {sector_key.replace('-', ' ').title()}")

    if overview is None:
        st.warning(f"No sector overview found. '{sector_key}'")
    else:
        st.write(
            f'**Market Weight**: {100*overview.get('market_weight', -9.99):.1f}%')

        st.write(
            f'**Description**: {overview.get("description", "No description available.")}')

    if top_companies is None:
        st.warning(f"No top companies found. '{sector_key}'")
    else:
        st.write("**Top Companies**")
        st.dataframe(top_companies)

    if top_etfs is None:
        st.warning(f"No top ETFs found. '{sector_key}'")
    else:
        # convert dict to pandas
        top_etfs_df = pd.DataFrame.from_dict(
            top_etfs, orient='index', columns=['ETF']).reset_index(names='Symbol')
        st.write("**Top ETFs**")
        st.dataframe(top_etfs_df, hide_index=True)
