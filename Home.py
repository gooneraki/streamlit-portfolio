# pylint: disable=C0103
""" Streamlit application for the Home page of the Portfolio Historical Data Analysis app. """
import streamlit as st

# Set the page title and layout
st.set_page_config(
    page_title="Portfolio Historical Data Analysis", layout="centered")

# Add a title
st.title("Welcome to Portfolio Historical Data Analysis")

# Add an introduction
st.write("""
This is a **personal finance analysis tool** built with Streamlit. 
It uses Python's `yfinance` library to fetch historical financial data directly from Yahoo Finance.
""")

# Features section
st.header("Features")
st.write("""
- **Single Asset Analysis**: Dive deep into the historical performance of a single asset and uncover key trends and statistics.
- **Portfolio Analysis**: Analyze the historical performance of your entire portfolio. Get comprehensive stats that provide insights into your portfolio's historical trends and overall performance.
""")

# Disclaimer section
st.warning("""
**Disclaimer**: While historical data can help forecast future trends, such predictions are inherently uncertain and should be used cautiously.  
This app is intended for informational purposes only and is not responsible for any financial losses incurred based on its use.
""")

# Footer or additional details
st.write("Explore the pages in the sidebar to start your analysis!")
