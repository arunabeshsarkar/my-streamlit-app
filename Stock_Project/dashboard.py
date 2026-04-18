import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import datetime

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ===== WHITE UI =====
st.markdown("""
<style>
.stApp {
    background-color: white;
    color: black;
}
h1, h2, h3, h4, h5, h6, p, span, label {
    color: black !important;
}
div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
}
button {
    background-color: #f0f2f6 !important;
    color: black !important;
}
table {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ===== TITLE =====
st.title("📈 Hybrid Stock & Crypto Prediction Dashboard")

# ===== STOCK LIST =====
stock_name = st.selectbox(
    "Select Stock / Crypto",
    [
        "AAPL", "TSLA", "GOOGL",

        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "KOTAKBANK.NS","SBIN.NS","AXISBANK.NS","LT.NS","ITC.NS",
        "HINDUNILVR.NS","BHARTIARTL.NS","ASIANPAINT.NS","MARUTI.NS",
        "SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS","NESTLEIND.NS",
        "BAJFINANCE.NS","BAJAJFINSV.NS","POWERGRID.NS","NTPC.NS",
        "ONGC.NS","COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS",
        "WIPRO.NS","HCLTECH.NS","TECHM.NS","INDUSINDBK.NS",
        "ADANIPORTS.NS","ADANIENT.NS","GRASIM.NS","CIPLA.NS",
        "DRREDDY.NS","APOLLOHOSP.NS","DIVISLAB.NS","EICHERMOT.NS",
        "HEROMOTOCO.NS","BAJAJ-AUTO.NS","BRITANNIA.NS","SHREECEM.NS",
        "UPL.NS","SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS",
        "TATACONSUM.NS","M&M.NS","HAVELLS.NS","DABUR.NS",

        "BTC-USD","ETH-USD","BNB-USD","SOL-USD",
        "XRP-USD","ADA-USD","DOGE-USD","MATIC-USD"
    ]
)

# ===== ASSET TYPE =====
if "-USD" in stock_name:
    st.write("Asset Type: Cryptocurrency")
else:
    st.write("Asset Type: Stock")

# ===== DATA =====
data = yf.download(stock_name, start="2018-01-01")

data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# ===== FEATURES =====
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
data['MACD'] = ta.trend.MACD(data['Close']).macd()

data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(5).std()
data['Momentum'] = data['Close'] - data['Close'].shift(5)

data['Volume_Change'] = data['Volume'].pct_change()
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)

# Target
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Clean
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# ===== CHART =====
st.subheader("📊 Price Chart")
st.line_chart(data[['Close','MA20','MA50']])

# ===== ML =====
features = [
    'RSI','MACD','MA20','MA50',
    'Returns','Volatility','Momentum',
    'Volume_Change','Lag1','Lag2'
]

X = data[features]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== MODELS =====
xgb = XGBClassifier(n_estimators=300, max_depth=4)
xgb.fit(X_scaled, y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# ===== PREDICTION =====
latest = scaler.transform(X.tail(1))

xgb_pred = xgb.predict(latest)
xgb_conf = xgb.predict_proba(latest)[0][xgb_pred[0]]

knn_pred = knn.predict(latest)
knn_conf = knn.predict_proba(latest)[0][knn_pred[0]]

# ===== LSTM =====
scaler_lstm = MinMaxScaler()
close_data = scaler_lstm.fit_transform(data[['Close']])

X_lstm, y_lstm = [], []

for i in range(10, len(close_data)):
    X_lstm.append(close_data[i-10:i])
    y_lstm.append(close_data[i])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

model_lstm = Sequential([
    LSTM(50, input_shape=(10,1)),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_lstm, y_lstm, epochs=5, verbose=0)

last_seq = close_data[-10:].reshape(1,10,1)
lstm_pred = model_lstm.predict(last_seq)
pred_price = scaler_lstm.inverse_transform(lstm_pred)[0][0]

# ===== DATE =====
tomorrow = datetime.date.today() + datetime.timedelta(days=1)

# ===== OUTPUT =====
st.subheader("📊 Tomorrow Prediction")
st.write("Date:", tomorrow)

if xgb_pred[0] == 1:
    st.success(f"XGBoost: 📈 UP (Confidence {xgb_conf:.2f})")
else:
    st.error(f"XGBoost: 📉 DOWN (Confidence {xgb_conf:.2f})")

if knn_pred[0] == 1:
    st.info(f"KNN: 📈 UP (Confidence {knn_conf:.2f})")
else:
    st.info(f"KNN: 📉 DOWN (Confidence {knn_conf:.2f})")

# Best Model
if xgb_conf > knn_conf:
    st.write("🏆 Best Model: XGBoost")
else:
    st.write("🏆 Best Model: KNN")

# LSTM
st.subheader("📉 LSTM Predicted Price")
st.write(f"Expected Price Tomorrow: {round(pred_price, 2)}")

# ===== DATA =====
st.subheader("📋 Latest Data")
st.write(data.tail())