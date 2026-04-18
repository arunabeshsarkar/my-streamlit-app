from moneycontrol_api import MoneyControl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

mc = MoneyControl()

stock_data = mc.get_stock("RELIANCE")
news_list = stock_data.get_news()

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

scores = []

for news in news_list[:10]:
    text = news['title']
    score = sid.polarity_scores(text)['compound']
    scores.append(score)
    print(text)

if scores:
    avg_sentiment = sum(scores)/len(scores)
else:
    avg_sentiment = 0

df = pd.DataFrame({"Sentiment":[avg_sentiment]})
df.to_csv("sentiment.csv", index=False)

print("Sentiment Score:", avg_sentiment)