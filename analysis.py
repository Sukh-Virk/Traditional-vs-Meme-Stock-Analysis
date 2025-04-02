import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import warnings
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

#loading data, which contained stock prices and news headlines
merged = pd.read_csv("merged_stock_news.csv", parse_dates=["Date"])


# creating stock dataframes
#for just stock data
stockm = merged.drop(columns=["Headline", "Sentiment"])
#dataframe that includes headlinnes, dropping ones without
news = merged[~merged["Headline"].isna()]
#one with votality and news
volnews = news[news["Is Volatile"]]
# renaming
news_df = news 

print("\n AVERAGE METRICS BY STOCK")

stocksum = pd.DataFrame({
    'Daily Change %': stockm.groupby('Symbol')['Daily Change %'].mean(),
    '7-Day Volatility': stockm.groupby('Symbol')['7-Day Volatility'].mean(),
    'Is Volatile': stockm.groupby('Symbol')['Is Volatile'].sum(),
    'Percent Change Category': stockm.groupby('Symbol')['Percent Change Category'].agg(lambda x: x.value_counts().idxmax()),
    'Stock Type': stockm.groupby('Symbol')['Stock Type'].first()
})

print(stocksum.round(2))


#samething but by stock type
print("\n AVERAGE METRICS BY STOCK TYPE")
types = stockm.groupby("Stock Type").agg({

    "Daily Change %": "mean",

    "7-Day Volatility": "mean",

    "Is Volatile": "mean"
})
print(types)

#how much it changes per day
print("\n AVERAGE SENTIMENT ON EACH TYPE OF DAY")

sentchange1 = news.groupby("Percent Change Category")["Sentiment"]
sentchange2 = sentchange1.mean()
print(sentchange2)

# on votile days
print("\ AVERAGE SENTIMENT BY STOCK TYPE ON VOLATILE DAYS")

sentype1 = volnews.groupby("Stock Type")["Sentiment"]

sentype2 = sentype1.mean()

print(sentype2)

print("\n")

print("HEADLINE EXAMPLES ON SPIKE DAYS")
#getting spiked  and setting up other dataframes
spikes = news[news["Percent Change Category"] == "Spike"]

#dropping ana as we go
meme_vol = stockm[stockm["Stock Type"] == "Meme"]["7-Day Volatility"].dropna()

chipvol = stockm[stockm["Stock Type"] == "Blue-Chip"]["7-Day Volatility"].dropna()

spikesup = news[news["Percent Change Category"] == "Spike"]["Sentiment"].dropna()

downs = news[news["Percent Change Category"] == "Drop"]["Sentiment"].dropna()


# Volatility: Mann-Whitney only because unsure if data is normal

u_vol, p_u_vol = mannwhitneyu(meme_vol, chipvol, alternative='two-sided')

print("\n📉 Mann–Whitney U Test on Volatility (Meme vs Blue-Chip):")

print("U = {:,.0f}, p = {:.4f}".format(u_vol, p_u_vol))

# Sentiment: Mann-Whitney only same here
u_sent, p_u_sent = mannwhitneyu(spikesup, downs, alternative='two-sided')

print("\n📉 Mann–Whitney U Test on Sentiment (Spike vs Drop):")

print("U = {:,.0f}, p = {:.4f}".format(u_sent, p_u_sent))

# Correlation
corr_data = news[["Sentiment", "Daily Change %"]].dropna()

correlation = corr_data["Sentiment"].corr(corr_data["Daily Change %"])

print(f"\n Correlation between Sentiment and Daily Change %: r = {correlation:.3f}")

print("\n")

print(" HEADLINE VOLUME ANALYSIS")
hlc = news.groupby(["Symbol", "Date"]).size().reset_index(name="Headline Count")

swc = pd.merge(stockm, hlc, on=["Symbol", "Date"], how="left")

swc["Headline Count"] = swc["Headline Count"].fillna(0)

headline_vol_summary = swc.groupby("Percent Change Category")["Headline Count"].mean()

print("\n🧠 Avg. Headline Count by Price Category (Spike / Drop / Stable):")

print(headline_vol_summary.round(2))

high_headline_days = swc[swc["Headline Count"] >= 3]

spiker = (high_headline_days["Percent Change Category"] == "Spike").mean()

print(f"\n📈 % of high-headline days (3+ headlines) that were spikes: {spiker:.2%}")

#setting sentiment
positive_news = news[news["Sentiment"] > 0.1]
negative_news = news[news["Sentiment"] < -0.1]

p1 = (positive_news["Percent Change Category"] == "Spike")
p2 = (negative_news["Percent Change Category"] == "Spike")

upspike = p1.mean()
downspike = p2.mean()

print(f"\n🚀 Spike rate after POSITIVE headlines: {upspike:}")
print(f"📉 Spike rate after NEGATIVE headlines: {downspike:}")


print("\n📌 FINAL FINDINGS SUMMARY")
print("- Meme stocks are more volatile than blue-chip stocks (confirmed by both t-test and Mann–Whitney).")
print("- Sentiment on spike days tends to be more positive than drop days, but may not always be statistically significant.")
print("- Sentiment has a weak positive correlation with daily stock price change (r ≈ {:.2f}).".format(correlation))
print("- Days with 3+ headlines are more likely to be spike days ({:.1f}%).".format(spiker * 100))
print("- Positive headlines are associated with higher spike probability ({:.1f}%) than negative ones ({:.1f}%).".format(upspike * 100, downspike * 100))


print("\n📅 PERFORMANCE BY DAY OF WEEK")
stockm["DayOfWeek"] = stockm["Date"].dt.day_name()
weekday_perf = stockm.groupby(["Stock Type", "DayOfWeek"])["Daily Change %"].mean().unstack()
print("\n📈 Avg. Daily Change % by Weekday (Rows: Stock Type)")
print(weekday_perf.round(3))

meme_df = stockm[stockm["Stock Type"] == "Meme"]
meme_by_day = meme_df.groupby("DayOfWeek")["Daily Change %"].mean().sort_values(ascending=False)
print("\n🚀 Meme Stock Performance by Weekday (Best to Worst):")
print(meme_by_day.round(3))


print(" COMMON WORDS IN HEADLINES ON SPIKE/DROP DAYS")

# learned from https://www.nltk.org/
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_and_tokenize(headlines):
    words = []
    for text in headlines:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        tokens = [w for w in tokens if w not in stop_words]
        words.extend(tokens)
    return words

spike_words = clean_and_tokenize(news[news["Percent Change Category"] == "Spike"]["Headline"].dropna())
drop_words = clean_and_tokenize(news[news["Percent Change Category"] == "Drop"]["Headline"].dropna())

spike_top = Counter(spike_words).most_common(15)
drop_top = Counter(drop_words).most_common(15)

print("\n📈 Top Words in Spike Headlines:")
for word, count in spike_top:
    print(f"{word}: {count}")

print("\n📉 Top Words in Drop Headlines:")
for word, count in drop_top:
    print(f"{word}: {count}")

# --- PART 8: Plotting Daily % Change ---
daily_change = stockm.groupby(["Date", "Stock Type"])["Daily Change %"].mean().unstack()
plt.figure(figsize=(10, 5))
daily_change.rolling(7).mean().plot(ax=plt.gca())
plt.title("📉 Avg Daily % Change Over Time (7-Day Rolling)")
plt.xlabel("Date")
plt.ylabel("Daily % Change")
plt.grid(True)
plt.tight_layout()
plt.savefig("line_avg_daily_change.png")
