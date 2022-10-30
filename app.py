# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:24:14 2022

@author: BHANUKIRAN
"""


import uvicorn
import pandas as pd
import streamlit as st
from datetime import date, timedelta
import nsepy as ns
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go




st.title('Stock Forecast App')

stocks = ('INFY', 'WIPRO', 'RELIANCE', 'TATAMOTORS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_months = st.slider('months of prediction:', 1, 12)
period = n_months * 30

@st.cache
def load_data(ticker):
    data = ns.get_history(symbol = "INFY", start = date(2012,10,1), end = date.today())
    data.drop(['Symbol', 'Series', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades', 
               'Deliverable Volume','%Deliverble'],  axis = 1, inplace = True)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
st.subheader('Time Series data with Rangeslider')

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
future_df = future[-278:]
forecast = m.predict(future_df)


st.subheader(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast, xlabel = 'Time', ylabel = 'Predictions')
st.plotly_chart(fig1)

forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast = forecast.rename(columns={"ds": "Date", "yhat": "Pred_Close", "yhat_lower" : "Pred_lower","yhat_upper":"Pred_upper"})
forecast['Date'] = forecast['Date'].apply(lambda x: x.date())


# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.head(30))

