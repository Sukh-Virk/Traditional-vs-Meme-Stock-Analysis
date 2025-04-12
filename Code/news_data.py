from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import time

newsapi = NewsApiClient(api_key='e59366a012b34dd283753fd40c81f7c7')


tickers = {
    'GME': 'GameStop OR GME',
    'BB': 'BlackBerry OR BB',
    'AAPL': 'Apple OR AAPL',
    'MSFT': 'Microsoft OR MSFT'
}


to_date = datetime.today().strftime('%Y-%m-%d')
from_date = (datetime.today() - timedelta(days=29)).strftime('%Y-%m-%d')

print(f"\n Pulling headlines from {from_date} to {to_date}\n")

combined_data = []

for symbol, query in tickers.items():
    print(f"🔍 Fetching articles for {symbol}...")

    try:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt',
            page_size=100,
            page=1
        )

        total = response['totalResults']
        articles = response['articles']

        for article in articles:
            combined_data.append({
                'Stock': symbol,
                'Published Date': article['publishedAt'][:10],
                'Headline': article['title'],
                'Total Article Count (Last 30 Days)': total
            })

        time.sleep(1)

    except Exception as e:
        print(f"Error for {symbol}: {e}")
        combined_data.append({
            'Stock': symbol,
            'Published Date': '',
            'Headline': f'ERROR: {e}',
            'Total Article Count (Last 30 Days)': 'ERROR'
        })

df_combined = pd.DataFrame(combined_data)
df_combined.to_csv("newsapi_last_30_days_combined.csv", index=False)
print(f"\nDone! Saved to newsapi_last_30_days_combined.csv")