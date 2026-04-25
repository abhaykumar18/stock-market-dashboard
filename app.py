import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------
# Page Config
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# -------------------------
# Load Data
df = pd.read_csv("stocks.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -------------------------
# Sidebar Filters
st.sidebar.title("🔍 Filters")

# Date filter
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())

df = df[(df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))]

# Column select
column = st.sidebar.selectbox("Select Column", df.columns)

# -------------------------
# Title
st.title("📊 Stock Market Professional Dashboard")

# -------------------------
# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(df['Close'].iloc[-1],2))
col2.metric("Highest Price", round(df['Close'].max(),2))
col3.metric("Lowest Price", round(df['Close'].min(),2))

# -------------------------
# Line Chart
st.subheader("📈 Price Trend")
st.line_chart(df['Close'])

# -------------------------
# Multi Column Chart
st.subheader("📊 Selected Column")
st.line_chart(df[column])

# -------------------------
# Moving Average
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()

st.subheader("📊 Moving Average")
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'], label='Close')
ax.plot(df['Date'], df['MA10'], label='MA10')
ax.plot(df['Date'], df['MA50'], label='MA50')
ax.legend()
st.pyplot(fig)

# -------------------------
# Histogram
st.subheader("📉 Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df['Close'], bins=30)
st.pyplot(fig2)

# -------------------------
# Candlestick (Simple)
st.subheader("🕯️ Candlestick Chart")

if all(col in df.columns for col in ['Open','High','Low','Close']):
    st.line_chart(df[['Open','High','Low','Close']])
else:
    st.warning("Candlestick data not available")

# -------------------------
# Correlation Heatmap
if 'Stock' in df.columns:
    st.subheader("🔥 Correlation Heatmap")
    pivot = df.pivot(index='Date', columns='Stock', values='Close')
    fig3, ax3 = plt.subplots()
    import seaborn as sns
    sns.heatmap(pivot.corr(), annot=True, ax=ax3)
    st.pyplot(fig3)

# -------------------------
# ML Prediction
st.subheader("🤖 Price Prediction (Linear Regression)")

df['Days'] = np.arange(len(df))
X = df[['Days']]
y = df['Close']

model = LinearRegression()
model.fit(X, y)

future_days = st.slider("Days to Predict", 1, 30, 5)

future_X = np.arange(len(df), len(df)+future_days).reshape(-1,1)
predictions = model.predict(future_X)

st.write("Predicted Prices:", predictions)

# Plot Prediction
fig4, ax4 = plt.subplots()
ax4.plot(df['Date'], df['Close'], label="Actual")
future_dates = pd.date_range(df['Date'].iloc[-1], periods=future_days+1)[1:]
ax4.plot(future_dates, predictions, label="Prediction")
ax4.legend()
st.pyplot(fig4)

# -------------------------
# Data Table
st.subheader("📄 Data")
st.dataframe(df)