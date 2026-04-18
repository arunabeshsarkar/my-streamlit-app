import yfinance as yf
import pandas as pd

stock = yf.download("AAPL", start="2022-01-01", end="2024-01-01")

stock['MA20'] = stock['Close'].rolling(20).mean()
stock['MA50'] = stock['Close'].rolling(50).mean()

stock.to_csv("stock_data.csv")

print("Done")