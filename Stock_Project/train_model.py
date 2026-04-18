import pandas as pd
import ta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load stock data
stock = pd.read_csv("stock_data.csv")

# 🔥 FIX 1: Convert columns to numeric
stock['Close'] = pd.to_numeric(stock['Close'], errors='coerce')
stock['MA20'] = pd.to_numeric(stock['MA20'], errors='coerce')
stock['MA50'] = pd.to_numeric(stock['MA50'], errors='coerce')

# 🔥 FIX 2: Remove missing values BEFORE indicators
stock = stock.dropna()

# Add technical indicators
stock['RSI'] = ta.momentum.RSIIndicator(stock['Close']).rsi()
stock['MACD'] = ta.trend.MACD(stock['Close']).macd()

# Create target (1 = UP, 0 = DOWN)
stock['Target'] = (stock['Close'].shift(-1) > stock['Close']).astype(int)

# Load sentiment data
sentiment = pd.read_csv("sentiment.csv")

# Add sentiment
stock['Sentiment'] = sentiment['Sentiment'][0]

# 🔥 FIX 3: Remove NaN after indicators
stock = stock.dropna()

# Features (inputs)
X = stock[['RSI', 'MACD', 'MA20', 'MA50', 'Sentiment']]

# Target (output)
y = stock['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# Predict tomorrow
latest = X.tail(1)
prediction = model.predict(latest)

if prediction[0] == 1:
    print("Prediction: Stock will go UP tomorrow")
else:
    print("Prediction: Stock will go DOWN tomorrow")