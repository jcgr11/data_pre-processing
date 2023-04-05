import yfinance as yf
import streamlit as st
import datetime
import pandas as pd
import pandas.tseries.offsets as offsets
import numpy as np
import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots
import plotly.graph_objs as go

init_notebook_mode(connected=True)
cf.go_offline()

@st.cache_data
def get_tickers():
    df = pd.read_csv("C:\py_scripts/mf&etf_tickers.csv")
    tickers = df['ticker'].tolist()
    return tickers

@st.cache_data
def load_data(symbol, start, end):
    # Get the previous business day before the start date
    start_date = pd.date_range(start - offsets.BDay(1), periods=1, freq='B')[0]
    
    # Get the next business day after the end date
    end_date = pd.date_range(end, periods=1, freq='B')[0] + offsets.BDay(1)
    
    data = yf.download(symbol, start_date, end_date, progress=False, auto_adjust=True)["Close"]

    # Calculate daily return
    daily_return = data.pct_change()

    return_plot_data = (data.pct_change())*100

    # Calculate geometric mean return
    geometric_mean = ((((1 + daily_return)).prod()) - 1) 

    return data, return_plot_data, geometric_mean

st.sidebar.header("Fund Parameters")

tickers = get_tickers()
ticker = st.sidebar.selectbox(
    "Ticker",
    tickers
)

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.date(2020, 1, 1)
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

st.title("A simple web app for calculating Mutual Fund & ETF returns")

st.write("""
    ### User Manual
    * Click the button at the top left corner of this web page.
    * Enter a Ticker, start date, & end date.
    * Press the 'Get Data' button below to display the fund Return & interactive data.
""")

if st.button("Get Data"):
    data, return_plot_data, geometric_mean = load_data(ticker, start_date, end_date)
    st.write(f"<b>{ticker} return from {start_date} to {end_date}: {geometric_mean:.4%}</b>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(
        go.Scatter(x=data.index, y=data.values, name='Adjusted Close Price'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=return_plot_data.index, y=return_plot_data.values, name='Daily Return'),
        row=2, col=1
    )

    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Adjusted Close Price', row=1, col=1)
    fig.update_yaxes(title_text='Daily Return', row=2, col=1)

    fig.update_layout(title=f"Daily {ticker} Price and Returns: {start_date} to {end_date}")

    st.plotly_chart(fig, use_container_width=True)

    data = data.reset_index() # move date and time to axis 1 index 0
    date = data['Date'].dt.date # remove time stamp
    ex_dt = data.iloc[:,1:] # create new date only index column
    data = ex_dt.set_index(date) # set date column
    col1, col2 = st.columns(2)
    with col1:
        st.write(data.rename(columns = {'Close':'Adj Close Price'}, inplace = False))
    with col2:
        st.write(data.rename(columns = {'Close':'Descriptive Statisticss'}, inplace = False).describe())

#streamlit run perf_calc.py