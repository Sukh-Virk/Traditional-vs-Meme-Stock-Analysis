from psaw import PushshiftAPI
import datetime as dt
import pandas as pd

#Initialize Pushshift API
api = PushshiftAPI()

#Parameters
stock_ticker = 'GME'
subreddit = 'wallstreetbets'
start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2021, 3, 1)

start_ts = int(start_date.timestamp())
end_ts = int(end_date.timestamp())

#Fetch posts
print(f"Fetching posts mentioning '{stock_ticker}' in r/{subreddit}...")
posts = api.search_submissions(q=stock_ticker,
                                subreddit=subreddit,
                                after=start_ts,
                                before=end_ts,
                                filter=['created_utc', 'title'],
                                limit=10000)

#Extract and structure the data
data = [{
    'date': dt.fromtimestamp(post.created_utc.timezone.utc).strftime('%Y-%m-%d'),
    'title': post.title
} for post in posts]

df = pd.DataFrame(data)
print(f"Collected {len(df)} posts.")