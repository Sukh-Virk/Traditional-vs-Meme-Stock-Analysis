#imports for data
import requests
import pandas as pd
from textblob import TextBlob
from datetime import datetime
import time

#api key for alpha vantage stock info
API_KEY = "9BHS3YUMT2RNCZ3F"  # Alpha Vantage key
BASE_URL = "https://www.alphavantage.co/query"

#our meme stocks chosen by team
memes = ["GME", "AMC", "BB"]
#our bluechip stocks
bs = ["AAPL", "MSFT", "JNJ"]
#merge all
alls = memes + bs

#function to classify daily change
def classify_change(pct):
    if pd.isna(pct):
        return None
    elif pct < -2:
        return "Drop"
    elif pct > 2:
        return "Spike"
    else:
        return "Stable"

#function for getting and cleaning stock data
def cleanstockd(symbol, stock_type):
    print(f"Fetching stock data for {symbol}...")

    #setting parameters for API request
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }

    #sending request to API
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    #checking if response contains expected data
    if "Time Series (Daily)" not in data:
        print(f"Error fetching {symbol}: {data}")
        return None

    #convert data to DataFrame
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    #rename columns to readable format
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    #convert numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #create new columns
    df["Daily Change %"] = df["Close"].pct_change() * 100
    df["7-Day Volatility"] = df["Daily Change %"].rolling(window=7).std()
    df["Is Volatile"] = df["7-Day Volatility"] > 5
    df["Percent Change Category"] = df["Daily Change %"].apply(classify_change)

    #adding identifiers
    df["Symbol"] = symbol
    df["Stock Type"] = stock_type
    df["Date"] = df.index

    return df.reset_index(drop=True)

#function to get stocktwits messages
def fetch_stocktwits_messages(symbol, total_messages=150):
    print(f"Fetching StockTwits messages for {symbol}...")

    messages = []
    max_id = None
    base_url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    headers = {"User-Agent": "Mozilla/5.0"}

    while len(messages) < total_messages:
        url = base_url
        if max_id:
            url += f"?max={max_id}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch StockTwits for {symbol}: {response.status_code}")
            break

        batch = response.json().get("messages", [])
        if not batch:
            break

        for msg in batch:
            body = msg.get("body", "")
            sentiment = TextBlob(body).sentiment.polarity
            messages.append({
                "Symbol": symbol,
                "Date": pd.to_datetime(msg["created_at"]).date(),
                "Headline": body,
                "Sentiment": sentiment
            })

        max_id = batch[-1]["id"] - 1
        time.sleep(1)

    return pd.DataFrame(messages[:total_messages])

#main function
if __name__ == "__main__":
    all_data = []
    all_news = []

    #get data for memes
    for mem in memes:
        df = cleanstockd(mem, "Meme")
        if df is not None:
            all_data.append(df)
        time.sleep(15)

    #get data for bluechips
    for b in bs:
        df = cleanstockd(b, "Blue-Chip")
        if df is not None:
            all_data.append(df)
        time.sleep(15)

    #get stocktwits messages
    for s in alls:
        news_df = fetch_stocktwits_messages(s, total_messages=150)
        if not news_df.empty:
            all_news.append(news_df)

    #combine stock data
    combined_stock_df = pd.concat(all_data, ignore_index=True)
    combined_stock_df["Date"] = pd.to_datetime(combined_stock_df["Date"])

    #combine news data
    if all_news:
        combined_news_df = pd.concat(all_news, ignore_index=True)
        combined_news_df["Date"] = pd.to_datetime(combined_news_df["Date"])
    else:
        print(" No StockTwits messages were fetched.")

        combined_news_df = pd.DataFrame(columns=["Symbol", "Date", "Headline", "Sentiment"])

    #merge both datasets by symbol and date
    merged_df = pd.merge(combined_stock_df, combined_news_df, on=["Symbol", "Date"], how="left")


    #save to CSV
    merged_df.to_csv("StockandSentiment.csv", index=False)

    print(" Dataset Saved")

