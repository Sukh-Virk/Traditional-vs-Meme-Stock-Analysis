#  Stock Price and Sentiment Analysis of Meme vs. Blue-Chip Stocks

This project explores stock prices of **meme stocks** such as GME, AMC, BB and compares them to **blue-chip stocks** such as AAPL, JNJ, and MSFT, analyzing the influence of sentiment through media. It also examines whether sentiment can predict stock movement.



##  Main Ideas

- Using APIs to gather financial and sentiment data  
- Performing basic operations such as averaging and summing  
- Conducting various statistical tests on the data  
- Applying different machine learning models for predictions  
- Applying NLP techniques to analyze sentiment  
- Reporting sentiment trends and findings  



## Requirements

This project uses **Python 3.8+** and the following libraries:

- `pandas` – for handling data and DataFrames  
- `requests` – for making API requests  
- `matplotlib` – for creating visuals and plots  
- `seaborn` – for improved visualization aesthetics  
- `textblob` – for sentiment analysis  
- `scipy` – for statistical testing  
- `nltk` – for stopword removal and lemmatization  
- `wordcloud` – for generating word clouds  
- `warnings` – for suppressing warning messages  
- `re`, `collections` – built-in Python modules for regex and word frequency analysis  
- `scikit Learn`  – built-in Python modules for regex and word frequency analysis
- `date`  – built-in Python modules for regex and word frequency analysis
- `datetime`  – built-in Python modules for regex and word frequency analysis
- `newsapi-python`  – built-in Python modules for regex and word frequency analysis




## Installation Guide

To run this project, you will need to install the following Python libraries individually using `pip`:

bash
pip install pandas
pip install requests
pip install textblob
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install wordcloud
pip install newsapi-python 


## How to Run the Project

To run this project follow these steps. First run `python Code/news_data.py` and `python Code/Stocktwits_data.py`. These files will get the csv files needed for analysis. Examples can be found under the Data folder. Then run `python Code/Sentiment_Analysis.py`, `python Code/models.py` to do some analysis on each stocktype. Lastly, run `python Code/models.py` to see predictions on and differences between meme stocks and tradtional stocks. These files will output visuals. Examples can be found under visuals folder.


## NLTK Downloads
These must be downloaded before using NLTK-based features:

python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")


 ## Contributors:
 Sukhman Virk 301468202
 Harsh Sidhu
Aniyah Bohnen