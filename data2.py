import requests
import pandas as pd
import feedparser
from textblob import TextBlob
from datetime import datetime
import time

API_KEY = "9BHS3YUMT2RNCZ3F" #my API key 
BASE_URL = "https://www.alphavantage.co/query" #link to endpoint
MEME_STOCKS = ["GME", "AMC", "BB"] #memstocks can change depending on  partners
BLUECHIP_STOCKS = ["AAPL", "MSFT", "JNJ"] #traditional ones
# all stocks, which include both
ALL_STOCKS = MEME_STOCKS + BLUECHIP_STOCKS


def classify_change(pct):
    if pd.isna(pct):
        return None
    elif pct < -2:
        return "Drop"
    elif pct > 2:
        return "Spike"
    else:
        return "Stable"

# --- Fetch and clean stock data ---
def fetch_and_clean_stock_data(symbol, stock_type):
    print(f"Fetching data for {symbol}...")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"Error fetching {symbol}: {data}")
        return None

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["Daily Change %"] = df["Close"].pct_change() * 100
    df["7-Day Volatility"] = df["Daily Change %"].rolling(window=7).std()
    df["Is Volatile"] = df["7-Day Volatility"] > 5
    df["Percent Change Category"] = df["Daily Change %"].apply(classify_change)
    df["Symbol"] = symbol
    df["Stock Type"] = stock_type
    df["Date"] = df.index

    return df.reset_index(drop=True)

# --- Fetch news headlines ---
def fetch_stock_news(symbol):
    print(f"Fetching news for {symbol}...")
    rss_url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    feed = feedparser.parse(rss_url)
    news_data = []

    for entry in feed.entries:
        title = entry.title
        published = datetime(*entry.published_parsed[:6])
        sentiment = TextBlob(title).sentiment.polarity

        news_data.append({
            "Symbol": symbol,
            "Date": published.date(),  # returns a date object
            "Headline": title,
            "Sentiment": sentiment
        })

    return pd.DataFrame(news_data)

# --- Main ---
if __name__ == "__main__":
    all_data = []
    all_news = []

    # Fetch stock data
    for symbol in MEME_STOCKS:
        df = fetch_and_clean_stock_data(symbol, "Meme")
        if df is not None:
            all_data.append(df)
        time.sleep(15)

    for symbol in BLUECHIP_STOCKS:
        df = fetch_and_clean_stock_data(symbol, "Blue-Chip")
        if df is not None:
            all_data.append(df)
        time.sleep(15)

    # Fetch news data
    for symbol in ALL_STOCKS:
        news_df = fetch_stock_news(symbol)
        all_news.append(news_df)

    # Combine data
    combined_stock_df = pd.concat(all_data, ignore_index=True)
    combined_news_df = pd.concat(all_news, ignore_index=True)

    combined_stock_df["Date"] = pd.to_datetime(combined_stock_df["Date"])
    combined_news_df["Date"] = pd.to_datetime(combined_news_df["Date"])

    # Merge on Symbol + Date
    merged_df = pd.merge(combined_stock_df, combined_news_df, on=["Symbol", "Date"], how="left")

    # Save to CSV
    merged_df.to_csv("merged_stock_news.csv", index=False)
    print("✅ Merged dataset saved as merged_stock_news.csv")
