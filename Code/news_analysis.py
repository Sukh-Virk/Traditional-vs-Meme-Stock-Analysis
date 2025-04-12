import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():

    stock_data = pd.read_csv("merged_stock_news.csv", parse_dates=["Date"])

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

    news_counts = news_data[['Stock', 'Total Article Count (Last 30 Days)']].drop_duplicates()

    # Join the summary with the news_counts
    combined_data = pd.merge(summary, news_counts, left_index=True, right_on='Stock', how='left').reset_index()
    combined_data['Total Article Count (Last 30 Days)'] = combined_data['Total Article Count (Last 30 Days)'].fillna(0)

    # TODO fix the index

    plt.figure()
    plt.scatter(combined_data['Total Article Count (Last 30 Days)'], combined_data['Spikes'])
    plt.xlabel('Total Article Count')
    plt.ylabel('Number of Spikes')
    plt.title('Correlation between Total News Articles and Stock Spikes')
    
    plt.figure()
    plt.scatter(combined_data['Total Article Count (Last 30 Days)'], combined_data['Drops'])
    plt.xlabel('Total Article Count')
    plt.ylabel('Number of Drops')
    plt.title('Correlation between Total News Articles and Stock Drops')
    

    #stocks_with_least_headlines = combined_data.sort_values(by='Total Article Count (Last 30 Days)').drop_duplicates()


    # plt.bar(stocks_with_least_headlines['Stock'], stocks_with_least_headlines['Stable'])
    # plt.xlabel('Stock Symbol')
    # plt.ylabel('Number of Stable Days')
    # plt.title('Stability of Stocks with Least Headlines')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    

    # Visualization
    # Setting the positions and width for the bars
    pos = list(range(len(combined_data['Spikes'])))
    width = 0.2  # width of a bar

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars for each data category
    plt.bar([p - width*1.5 for p in pos], combined_data['Spikes'], width, alpha=0.6, color='b', label='Spikes')
    plt.bar([p - width*0.5 for p in pos], combined_data['Drops'], width, alpha=0.6, color='r', label='Drops')
    plt.bar([p + width*0.5 for p in pos], combined_data['Stable'], width, alpha=0.6, color='g', label='Stable')
    #plt.bar([p + width*1.5 for p in pos], combined_data['Total Article Count (Last 30 Days)'], width, alpha=0.6, color='y', label='Headlines')

    # Setting labels and titles
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Stock Behavior')
    ax.set_xticks([p for p in pos])
    ax.set_xticklabels(combined_data['Stock'])
    plt.xticks(rotation=90)  # Rotate labels to ensure they fit and are readable

    # Adding the legend and showing the plot
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(combined_data)       



if __name__=='__main__':
    main()