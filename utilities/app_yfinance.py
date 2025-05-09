import yfinance as yf
import streamlit as st


@st.cache_data
def get_sector_details(sector_name: str):
    """ Get the sector info """
    if sector_name in yf.const.SECTOR_INDUSTY_MAPPING:
        sector = yf.Sector(sector_name)
        return {'sector_name': sector_name, 'overview': sector.overview, 'top_companies': sector.top_companies, 'top_etfs': sector.top_etfs}


@st.cache_data
def retrieve_sector_industry_keys():
    """ Get the sector and industry keys """
    return list(yf.const.SECTOR_INDUSTY_MAPPING.keys())
