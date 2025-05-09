import yfinance as yf
import streamlit as st


@st.cache_data
def get_sector_details(sector_name: str):
    """ Get the sector info """
    try:
        sector = yf.Sector(sector_name)
        return ({
            'sector_name': sector_name,
            'overview': sector.overview,
            'top_companies': sector.top_companies,
            'top_etfs': sector.top_etfs})
    except Exception as err:
        error_message = f"Error retrieving sector details for '{sector_name}': {err}"
        return error_message


@st.cache_data
def retrieve_sector_industry_keys():
    """ Get the sector and industry keys """
    return list(yf.const.SECTOR_INDUSTY_MAPPING.keys())
