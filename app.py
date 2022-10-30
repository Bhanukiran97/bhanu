# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 09:21:25 2022

@author: BHANUKIRAN
"""


import uvicorn
import pandas as pd
import streamlit as st
from datetime import date
import nsepy as ns
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go




now = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('INFY', 'WIPRO', 'RELIANCE', 'TATAMOTORS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 2)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = ns.get_history(symbol = "INFY", start = date(2012,10,1), end = now)
    data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
               'Deliverable Volume','%Deliverble'],  axis = 1, inplace = True)
    pd.DatetimeIndex(data.index, inplace = True)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.head())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)