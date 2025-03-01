import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def fetch_stock_data(symbol, period='1y'):
    stock = yf.Ticker(symbol + ".NS")  # NSE symbol
    df = stock.history(period=period)
    return df

def plot_stock_price(df, symbol):
    st.subheader(f"Stock Price Trend for {symbol}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)

def calculate_moving_average(df, window=20):
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def predict_future_prices(df, forecast_days=30):
    df = df.dropna()
    df['Day'] = np.arange(len(df)).reshape(-1, 1)
    
    X = df[['Day']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    
    return future_days.flatten(), future_predictions

# Streamlit App
st.title("📈 NSE Stock Price Prediction")

# User Input for Stock Symbol
symbol = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE, TCS, INFY):").upper()

if symbol:
    try:
        df = fetch_stock_data(symbol)
        
        if df.empty:
            st.error("Invalid stock symbol or no data available.")
        else:
            # Display raw data
            st.write("### Stock Data Preview")
            st.dataframe(df.tail())
            
            # Plot Stock Price
            plot_stock_price(df, symbol)
            
            # Moving Average
            df = calculate_moving_average(df)
            st.write("### Moving Average (SMA 20 Days)")
            st.line_chart(df[['Close', 'SMA_20']])
            
            # Stock Statistics
            st.write("### Stock Statistics")
            st.write(df.describe())
            
            # Future Price Prediction
            st.write("### Future Price Prediction (Next 30 Days)")
            future_days, future_prices = predict_future_prices(df)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
            ax.plot(pd.date_range(df.index[-1], periods=30, freq='D'), future_prices, label='Predicted Prices', linestyle='dashed', color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error fetching data: {e}")