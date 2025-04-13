import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy import stats

def main():

    stock_data = pd.read_csv("StockandSentiment.csv", parse_dates=["Date"])

    stocks = stock_data[["Percent Change Category","Symbol", "Date"]]

    count_spikes = stocks[stocks['Percent Change Category'] == 'Spike'].groupby('Symbol').size()
    count_stable = stocks[stocks['Percent Change Category'] == 'Stable'].groupby('Symbol').size()
    count_drops = stocks[stocks['Percent Change Category'] == 'Drop'].groupby('Symbol').size()

    # Combine the counts into a single DataFrame for easier analysis
    summary = pd.DataFrame({
        'Spikes': count_spikes,
        'Stable': count_stable,
        'Drops': count_drops
    })

    # Fill NaN values with 0, assuming no occurrences were NaN means there were no events of that type
    summary = summary.fillna(0)

    news_data = pd.read_csv("newsapi_last_30_days_combined.csv", parse_dates=["Published Date"])

    # Since it has the same count for the same stock (total) we dont want to add multiple totals
    news_counts = news_data[['Stock', 'Total Article Count (Last 30 Days)']].drop_duplicates()

    # Join the summary with the news_counts
    combined_data = pd.merge(summary, news_counts, left_index=True, right_on='Stock', how='left').reset_index()
    combined_data['Total Article Count (Last 30 Days)'] = combined_data['Total Article Count (Last 30 Days)'].fillna(0)

    combined_data = combined_data.drop(columns=['index'])

    # Plot the data
    plt.figure()
    plt.scatter(combined_data['Total Article Count (Last 30 Days)'], combined_data['Spikes'])
    plt.scatter(combined_data['Total Article Count (Last 30 Days)'], combined_data['Drops'])
    plt.scatter(combined_data['Total Article Count (Last 30 Days)'], combined_data['Stable'])
    plt.xlabel('Total Article Count')
    plt.ylabel('Count of Spikes/Drops/Stable Days')

    plt.legend(['Spikes', 'Drops', 'Stable'], loc=1)
    
    plt.title('Correlation Between News Headlines and Stock Behaviour')
    plt.savefig('headlines-stocks.jpeg', format='jpeg')

    # Creating Bar Graph
    # Setting the positions and width for the bars
    pos = list(range(len(combined_data['Spikes'])))
    width = 0.2  # width of a bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars for each data category
    plt.bar([p - width*1.5 for p in pos], combined_data['Spikes'], width, alpha=0.6, color='b', label='Spikes')
    plt.bar([p - width*0.5 for p in pos], combined_data['Drops'], width, alpha=0.6, color='r', label='Drops')
    plt.bar([p + width*0.5 for p in pos], combined_data['Stable'], width, alpha=0.6, color='g', label='Stable')
    #plt.bar([p + width*1.5 for p in pos], math.log(combined_data['Total Article Count (Last 30 Days)']), width, alpha=0.6, color='y', label='Headlines')

    
    ax.set_ylabel('Count')
    ax.set_xlabel('Stock Name')
    ax.set_title('Individual Comparison of Stock Behavior')
    ax.set_xticks([p for p in pos])
    ax.set_xticklabels(combined_data['Stock'])
    plt.xticks(rotation=90)  # Rotate labels to ensure they fit and are readable

    # Adding the legend and showing the plot
    plt.legend(loc=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('news-bar-plot.jpeg', format='jpeg')
    

    #print(combined_data, "\n")       

    # Linear regression between headlines and each behavior (spikes, drops, and stable days)
    spike_reg = stats.linregress(combined_data['Total Article Count (Last 30 Days)'], combined_data['Spikes'])
    drop_reg = stats.linregress(combined_data['Total Article Count (Last 30 Days)'], combined_data['Drops'])
    stable_reg = stats.linregress(combined_data['Total Article Count (Last 30 Days)'], combined_data['Stable'])

    print("\nSpikes Regression:")
    print(f"  Slope: {spike_reg.slope:.3f}, Intercept: {spike_reg.intercept:.3f}")
    print(f"  R-value: {spike_reg.rvalue:.3f}, R-squared: {spike_reg.rvalue**2:.3f}")

    print("\nDrops Regression:")
    print(f"  Slope: {drop_reg.slope:.3f}, Intercept: {drop_reg.intercept:.3f}")
    print(f"  R-value: {drop_reg.rvalue:.3f}, R-squared: {drop_reg.rvalue**2:.3f}")

    print("\nStable Regression:")
    print(f"  Slope: {stable_reg.slope:.3f}, Intercept: {stable_reg.intercept:.3f}")
    print(f"  R-value: {stable_reg.rvalue:.3f}, R-squared: {stable_reg.rvalue**2:.3f}\n")


if __name__=='__main__':
    main()