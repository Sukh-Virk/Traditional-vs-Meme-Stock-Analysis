#imports by need for project

import pandas as pd

from scipy.stats import mannwhitneyu
#import for getting warnings
import warnings
#for graphs
import matplotlib.pyplot as plt
#import to make  grahs look better
import seaborn as sns
#import to keep counter
from collections import Counter
#regex
import re
#imports for nlp
import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
#wordcloud visual
from wordcloud import WordCloud


warnings.filterwarnings("ignore")
#words for stopping
nltk.download("stopwords")
nltk.download("wordnet")

#load our data we will use
merg = pd.read_csv("StockandSentiment.csv", parse_dates=["Date"])
#we drop the columns where we do  not need empty rows in the headline and sentiment part
merg = merg.dropna(subset=["Headline", "Sentiment"])

#for this part we only look at non sentiment data
stockm = merg.drop(columns=["Headline", "Sentiment"])
news = merg
volnews = news[news["Is Volatile"]]

print("\n average metric per stock")

#daily change
dailyc = stockm.groupby("Symbol")["Daily Change %"].mean()

# 7-day volatility mean
vola = stockm.groupby("Symbol")["7-Day Volatility"].mean()

#total days
volc = stockm.groupby("Symbol")["Is Volatile"].sum()

# Most common percent change category
tp = stockm.groupby("Symbol")["Percent Change Category"].agg(lambda x: x.value_counts().idxmax())

# First stock type seen
st = stockm.groupby("Symbol")["Stock Type"].first()

#put into pd data frame
stocksum = pd.DataFrame({
    "Daily Change %": dailyc,
    "7-Day Volatility": vola,
    "Is Volatile": volc,
    "Percent Change Category": tp,
    "Stock Type": st
})

#we print the dataframe
print(stocksum.round(2))

print("\n")
#metric by stock type
print("Average Metric Per Stock")
#aggreate and get the avg per stock type
types = stockm.groupby("Stock Type").agg({
    "Daily Change %": "mean",
    "7-Day Volatility": "mean",
    "Is Volatile": "mean"
})
#print
print(types)

print("\n")
print("Average Sentiment On Each time")
print(news.groupby("Percent Change Category")["Sentiment"].mean())

print("\n")

print("Average Sentiment By Stock Type on Volatile Days")
outp = volnews.groupby("Stock Type")["Sentiment"].mean()
print(outp)


#setting stock type as mem
memem = stockm["Stock Type"] == "Meme"
#memedata 
memed = stockm[memem]
#meme vol but dropping empty
meme_vol = memed["7-Day Volatility"].dropna()

#we will do the same for the rest blue chip
chipvolm = stockm["Stock Type"] == "Blue-Chip"
chipvold = stockm[chipvolm]
chipv= chipvold["7-Day Volatility"].dropna()
#for spikes
spikem= news["Percent Change Category"] == "Spike"
spiked = news[spikem]
spikesup = spiked["Sentiment"].dropna()

#for drops
dropm = news["Percent Change Category"] == "Drop"
dropd = news[dropm]
downs = dropd["Sentiment"].dropna()


#doing manwhitney test on data because not sure
#this will tell us if the two groups differ signifcantly

result = mannwhitneyu(meme_vol, chipv, alternative='two-sided')

# Extract U statistic and p-value from the result
uv = result[0]
pv = result[1]

# Print the results simply
print("\n")
print("Mann–Whitney U Test on Volatility (Meme vs Blue-Chip):")
print("U =", uv, "and p =", pv)

#doing the same but for drops vs spikes

results = mannwhitneyu(spikesup, downs, alternative='two-sided')

# Extract U statistic and p-value
us = results[0]
pus = results[1]

# Print the results in a simple way
print("\nMann–Whitney U Test on Sentiment (Spike vs Drop):")

print("U =", us, "and p =", pus)

#checking the correlction
print("\n")
cor = news[["Sentiment", "Daily Change %"]].dropna()
correlation = cor["Sentiment"].corr(cor["Daily Change %"])
print("Correlation between Sentiment and Daily Change %: r = ")
print(str(correlation))

#headlines
hlc = news.groupby(["Symbol", "Date"]).size().reset_index(name="Headline Count")
#merging
swc = pd.merge(stockm, hlc, on=["Symbol", "Date"], how="left")
#fill empty with 0
swc["Headline Count"] = swc["Headline Count"].fillna(0)

headline_vol_summary = swc.groupby("Percent Change Category")["Headline Count"].mean()
print("\n Avg. Headline Count by Price Category (Spike / Drop / Stable):")
print(headline_vol_summary.round(2))

hhd = swc[swc["Headline Count"] >= 3] #with more than or equal to 3

spiker = (hhd["Percent Change Category"] == "Spike").mean()
print("\n% of high-headline days (3+ headlines) that were spikes: " )
print(str(spiker))

#doing sentiment analysis-
posnews = news[news["Sentiment"] > 0.1]
nnews = news[news["Sentiment"] < -0.1]
upspike = (posnews["Percent Change Category"] == "Spike").mean()
downspike = (nnews["Percent Change Category"] == "Spike").mean()

print("\nSpike rate after POSITIVE headlines: {:.2%}".format(upspike))
print("Spike rate after NEGATIVE headlines: {:.2%}".format(downspike))

#lets check which type of stocks do better
print("\n")
print("Performance Days by Week")
stockm["DayOfWeek"] = stockm["Date"].dt.day_name()
weekday_perf = stockm.groupby(["Stock Type", "DayOfWeek"])["Daily Change %"].mean().unstack()
print("\n")
print(" Avg. Daily Change % by Weekday (Rows: Stock Type)")
print(weekday_perf.round(3))

meme_df = stockm[stockm["Stock Type"] == "Meme"]
meme_by_day = meme_df.groupby("DayOfWeek")["Daily Change %"].mean().sort_values(ascending=False)
print("\n")
print("Meme Stock Performance by Weekday (Best to Worst):")
print(meme_by_day.round(3))


print("\n Sentiment Reaction Comparison by Stock Type")

# Separate positive and negative sentiment rows
pos_news = news[news["Sentiment"] > 0.1]
neg_news = news[news["Sentiment"] < -0.1]

# Group by stock type and calculate avg price change
pos_reaction = pos_news.groupby("Stock Type")["Daily Change %"].mean()
neg_reaction = neg_news.groupby("Stock Type")["Daily Change %"].mean()

# Print the comparison
print("\n Average Daily % Change After POSITIVE Sentiment:")
print(pos_reaction.round(3))

print("\n Average Daily % Change After NEGATIVE Sentiment:")
print(neg_reaction.round(3))


#NLP analysis
def nlps (headlines):
    words = []
    stop_words = set(stopwords.words("english"))
    stop_words.update([
        "https", "http", "com", "amp", "stock", "stocks", "buy", "market", "jnj", "mfst", "aapl", "gme",
        "rt", "via", "news", "share", "shares", "price", "today", "week", "now"
    ])
    lemmatizer = WordNetLemmatizer()

    #for each headline
    for text in headlines:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and not w.startswith("$")]
        words.extend(tokens)
    return words

print("\n")
print(" Most Common Words in Spike and Drop days")
spike_words = nlps(news[news["Percent Change Category"] == "Spike"]["Headline"].dropna())
drop_words = nlps(news[news["Percent Change Category"] == "Drop"]["Headline"].dropna())

#choose top 15 most common drops and spike words
spike_top = Counter(spike_words).most_common(15)
drop_top = Counter(drop_words).most_common(15)

print("\nTop Words in Spike Headlines:")
for word, count in spike_top:
    print(word, "-", count)


print("\nTop Words in Drop Headlines:")
for word, count in drop_top:
    print(word, "-", count)


print ("\n")

print("Polarity Per Stock")


def sl(score):
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

news["Sentiment_Label"] = news["Sentiment"].apply(sl)


sentiment_counts = news.groupby(["Symbol", "Sentiment_Label"]).size()

sentsummary = sentiment_counts.unstack(fill_value=0)


print(sentsummary)


#dis words
spike_freq = Counter(spike_words)
drop_freq = Counter(drop_words)
all_words = set(spike_freq.keys()).union(set(drop_freq.keys()))
discriminative = []

for word in all_words:
    spike_count = spike_freq[word]
    drop_count = drop_freq[word]
    total = spike_count + drop_count
    if total > 0:
        discriminative.append((word, spike_count / total, drop_count / total))

discriminative_sorted = sorted(discriminative, key=lambda x: abs(x[1] - x[2]), reverse=True)[:15]

print("\n Words Most Associated With Spikes or Drops:")
for word, spike_ratio, drop_ratio in discriminative_sorted:
    tag = "Spike" if spike_ratio > drop_ratio else "Drop"
    print(f"{word}: {tag}-leaning ({spike_ratio:.2%} spike, {drop_ratio:.2%} drop)")

    # Create DataFrame to compare reactions
reaction_df = pd.DataFrame({
    "Positive": pos_reaction,
    "Negative": neg_reaction
}).T


reaction_df.plot(kind="bar", figsize=(8, 5))
plt.title("Price Reaction to Sentiment by Stock Type")
plt.ylabel("Average Daily % Change")
plt.xlabel("Sentiment Type")
plt.grid(axis='y')
plt.legend(title="Stock Type")
plt.tight_layout()
plt.savefig("sentiment_impact_by_type.png")







# Generate word cloud from spike words
spike_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(Counter(spike_words))

plt.figure(figsize=(10, 5))
plt.imshow(spike_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Common Words in Spike Headlines")
plt.tight_layout()
plt.savefig("spike_wordcloud.png") 


print(f"Word cloud saved")




