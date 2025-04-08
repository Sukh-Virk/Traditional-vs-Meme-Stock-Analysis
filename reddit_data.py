
# In order to request a temp auth token from reddit

with open('api_pw.txt', 'r') as f:
    SECRET_KEY = f.read()

CLIENT_ID = 'tCA_H1FzGCBfUTWfOa_IvA'

import requests
import pandas as pd

auth = requests.auth.HTTPBasicAuth(CLIENT_ID , SECRET_KEY)

with open('r_pw.txt', 'r') as f:
    pw = f.read()

data = {
    'grant_type': 'password',
    'username': 'Master-County9017',
    'password': pw
}

headers = {'User-Agent': 'MyAPI/0.0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

TOKEN = res.json()['access_token']

headers['Authorization'] = f'bearer {TOKEN}'

# Gets 100 data items from the 'stocks' subreddit on the 'new' page
res = requests.get('https://oauth.reddit.com/r/stocks/new',
                   headers=headers, params={'limit': '100'})

res.json()

posts = []

for post in res.json()['data']['children']:
    posts.append({
        'subreddit': post['data']['subreddit'],
        'title': post['data']['title'],
        'selftext': post['data']['selftext'],
        'upvote_ratio': post['data']['upvote_ratio'],
        'score': post['data']['score'],
        'created_utc': post['data']['created_utc']
    })

df = pd.DataFrame(posts)

# To view the possible items we can add for data filtering
#print(post['data'].keys())

#print(df)

df.to_csv("reddit-data.csv", index=False)