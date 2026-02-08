# ======================
# app.py (Fixed & Enhanced)
# ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# =======================
# Load Datasets
# =======================
@st.cache_data
def load_data():
    btc = pd.read_csv("cleaned_btc_data.csv")
    eth = pd.read_csv("cleaned_eth_data.csv")
    doge = pd.read_csv("cleaned_doge_data.csv")

    btc_arima = pd.read_csv("btc_arima_forecast.csv")
    btc_sarima = pd.read_csv("btc_sarima_forecast.csv")
    btc_lstm = pd.read_csv("btc_lstm_forecast.csv")
    btc_prophet = pd.read_csv("btc_prophet_forecast_30days.csv")

    eth_arima = pd.read_csv("eth_arima_forecast.csv")
    eth_sarima = pd.read_csv("eth_sarima_forecast.csv")
    eth_lstm = pd.read_csv("eth_lstm_forecast.csv")
    eth_prophet = pd.read_csv("eth_prophet_forecast.csv")

    doge_arima = pd.read_csv("doge_arima_forecast.csv")
    doge_sarima = pd.read_csv("doge_sarima_forecast.csv")
    doge_lstm = pd.read_csv("doge_lstm_forecast.csv")
    doge_prophet = pd.read_csv("doge_prophet_forecast.csv")

    model_eval = pd.read_csv("model_evaluation_summary.csv")

    # =======================
    # Normalize column names
    # =======================
    forecast_dfs = [btc_arima, btc_sarima, btc_lstm, btc_prophet,
                    eth_arima, eth_sarima, eth_lstm, eth_prophet,
                    doge_arima, doge_sarima, doge_lstm, doge_prophet]

    for df in forecast_dfs:
        # Convert date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Rename possible prediction columns
        rename_map = {
            'Forecast': 'Predicted_Close',
            'Predicted': 'Predicted_Close',
            'yhat': 'Predicted_Close',
            'y_pred': 'Predicted_Close'
        }
        df.rename(columns=rename_map, inplace=True)

        # Rename actual if needed
        if 'Actual' not in df.columns and 'Close' in df.columns:
            df.rename(columns={'Close': 'Actual'}, inplace=True)

    return btc, eth, doge, btc_arima, btc_sarima, btc_lstm, btc_prophet, \
           eth_arima, eth_sarima, eth_lstm, eth_prophet, \
           doge_arima, doge_sarima, doge_lstm, doge_prophet, model_eval


# Load all data
btc, eth, doge, btc_arima, btc_sarima, btc_lstm, btc_prophet, \
eth_arima, eth_sarima, eth_lstm, eth_prophet, \
doge_arima, doge_sarima, doge_lstm, doge_prophet, model_eval = load_data()


# =======================
# Sidebar Menu
# =======================
st.sidebar.title("📊 Crypto Dashboard")
menu = st.sidebar.radio("Navigate:", ["Overview", "Data View", "EDA", "Forecasts", "Model Evaluation"])


# =======================
# Overview
# =======================
if menu == "Overview":
    st.title("📈 Cryptocurrency Analysis Dashboard")
    st.markdown("""
    *Project:* Cryptocurrency EDA, Forecasting, and Model Evaluation
    *Dataset Source:* Yahoo Finance (yfinance)
    *Coins Analyzed:* BTC, ETH, DOGE


    *Work Done So Far:*Data Cleaning & Preprocessing:
                Removed missing values, handled outliers, formatted timestamps, and prepared data for modeling.
                Exploratory Data Analysis (EDA):
                Visualized and analyzed price trends, volatility, and trading volume patterns.
                Forecasting Models Implemented:
                ARIMA
                SARIMA
                LSTM (Long Short-Term Memory)
                Prophet

    *Model Evaluation:*Created a comparison table showing performance metrics for all models across all three cryptocurrencies.
    """)
    st.image("https://picsum.photos/800/400", caption="Test Image", use_container_width=True)





# =======================
# Data View
# =======================
elif menu == "Data View":
    st.title("💾 Data View")
    coin = st.selectbox("Select Cryptocurrency:", ["BTC", "ETH", "DOGE"])
    if coin == "BTC":
        st.dataframe(btc.head(50))
    elif coin == "ETH":
        st.dataframe(eth.head(50))
    else:
        st.dataframe(doge.head(50))


# =======================
# EDA
# =======================
elif menu == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    coin = st.selectbox("Select Cryptocurrency for EDA:", ["BTC", "ETH", "DOGE"])

    df = btc if coin == "BTC" else eth if coin == "ETH" else doge

    st.subheader("Closing Price Trend")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], color='blue')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Price Distribution")
    fig2 = px.histogram(df, x='Close', nbins=50, color_discrete_sequence=['orange'])
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig3, ax3 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    if 'Volume' in df.columns:
        st.subheader("Volume vs Price Scatter")
        fig4 = px.scatter(df, x='Volume', y='Close', color='Close', size='Volume')
        st.plotly_chart(fig4)


# =======================
# Forecasts (Fixed)
# =======================
elif menu == "Forecasts":
    st.title("📊 Forecast Models")
    coin = st.selectbox("Select Cryptocurrency:", ["BTC", "ETH", "DOGE"])
    model = st.selectbox("Select Model:", ["ARIMA", "SARIMA", "LSTM", "PROPHET"])

    forecast_map = {
        "BTC": {"ARIMA": btc_arima, "SARIMA": btc_sarima, "LSTM": btc_lstm, "PROPHET": btc_prophet},
        "ETH": {"ARIMA": eth_arima, "SARIMA": eth_sarima, "LSTM": eth_lstm, "PROPHET": eth_prophet},
        "DOGE": {"ARIMA": doge_arima, "SARIMA": doge_sarima, "LSTM": doge_lstm, "PROPHET": doge_prophet}
    }

    df_forecast = forecast_map[coin][model]
    st.dataframe(df_forecast.head(50))

    if 'Date' in df_forecast.columns and 'Predicted_Close' in df_forecast.columns:
        # Combined chart (Actual + Predicted)
        if 'Actual' in df_forecast.columns:
            fig = px.line(df_forecast, x='Date', y=['Actual', 'Predicted_Close'],
                          title=f"{coin} {model} Forecast (Actual vs Predicted)")
        else:
            fig = px.line(df_forecast, x='Date', y='Predicted_Close',
                          title=f"{coin} {model} Forecast")
        st.plotly_chart(fig)
    else:
        st.warning("⚠ Forecast data missing required columns (Date, Predicted_Close). Please check the CSV file format.")


# =======================
# Model Evaluation
# =======================
elif menu == "Model Evaluation":
    st.title("🏆 Model Evaluation Summary")
    st.dataframe(model_eval)
