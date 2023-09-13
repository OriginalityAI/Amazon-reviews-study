import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from prophet import Prophet
import requests
import json
from tqdm.auto import tqdm
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import scienceplots
milestone_date = pd.Timestamp("2022-11-30")


def get_data_for_analysis():
    """
    gets the processed data
    updates column types
    """
    df = pd.read_csv('Data/data_for_analysis.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['ratingScore'] = df['ratingScore'].astype('int').reset_index(drop=True)
    return df

def get_raw_data():
    """returns source data with sensitive information removed"""
    import pandas as pd
    return pd.read_csv('Data/raw_desensitized_data.csv')

def generate_helpfulScore(helpfulCount, totalCategoryRatings, totalCategoryReviews):
    """
    uses totalCategoryRatings (int) and totalCategoryReviews (int) to standardize the helpfulCount
    and returns a helpfulScore (float) between 0 and 1.
    """
    # gets the geometric mean of totalReviews and totalRatings or one if the other is 0
    geo_mean = totalCategoryReviews if totalCategoryRatings == 0 else \
        totalCategoryRatings if totalCategoryReviews == 0 else \
        math.sqrt((totalCategoryRatings*totalCategoryReviews))

    helpfulScore = helpfulCount/geo_mean
    return min(1, helpfulScore)

def plot_trend(df, start=2022, freq='1Y'):
    """
    turns the ai_content into a timeseries and plots it
    from the given start date
    and given timeseries frequency
    """
   
    data = df.copy()
    data.rename(columns={'ai_content': 'aiContent'})
#     data[data['ai_content'].map(type) !=float] check for errors
    tmp_ = data[['aiContent', 'date']].groupby(
        pd.Grouper(key="date", freq='1D')).mean()
    tmp_ = tmp_.dropna()['aiContent'][str(start):]
 
    tmp = data[['aiContent', 'date']].groupby(
        pd.Grouper(key="date", freq=freq)).mean()
    tmp = tmp.dropna()['aiContent'][str(start):]
    
    with plt.style.context(['science', 'ieee', 'high-vis']):

        tmp.plot(kind='bar', label='aiContent')
        plt.title('AI generated Amazon reviews')
        plt.yticks(np.arange(0,0.1,1))
        plt.xticks(range(len(tmp.index)), tmp.index.strftime('%Y'))
        
        # Color difference between before and after Nov 2022
        for idx, date in enumerate(tmp.index):
            if date > milestone_date:
                plt.gca().patches[idx].set_facecolor('orange')
        
        # # Add labels
        # for container in plt.containers:
        #     plt.bar_label(container, fmt='%.3f', color='brown', padding=1.5)

        for i, val in enumerate(tmp):
            plt.text(i, val, f'{val:.3f}', ha='center', va='bottom', color='brown')    
        plt.show()

    return tmp_

def missing_values(data):
    """
    function to fill missing values based on condition
    """
    split = '2022-11-30'
    df1, df2 = data[:split], data[split:]
    return pd.concat([df1.fillna(0), df2.interpolate('nearest')])

def generate_df_plot(tmp, start='2022-01'):
    """
    converts df to a timeseries with a dateindex and filled missing values
    """
    time_df = plot_trend(tmp, start)
    # reindex the time series
    date_index = pd.date_range(start=time_df.index[0],
                               end=time_df.index[-1],
                               freq='D') # D
    time_df = time_df.reindex(date_index).astype(float)
    # time_df = time_df.interpolate('nearest')
    time_df = missing_values(time_df)
    return time_df

def categorical_testing(df, col, col_2='aiContent'):
    """visualize and stats test for dependencies between 
    input: col a categorical feature and col_2, a numerical feature
    output: barplot and analysis result
    """
    import seaborn as sns
    from scipy.stats import f_oneway

    # bar plot
    with plt.style.context(['science', 'std-colors']):
        sns.barplot(data=df, x=col, y=col_2)
        plt.xticks(rotation=90)
        plt.yticks([])
        plt.title(f'Visual analysis of {col} vs {col_2}')
        # plt.legend()
        plt.grid()
        plt.show()

    # f-oneway stats test
    catgroup = df.groupby(col)[col_2].apply(list)
    p = f_oneway(*catgroup)[1]
    print(f"p value is {p:05f}.")
    print(f"{col_2} and {col} are",
          "correlated." if p < 0.05 else "not correlated.\n\n")

def numerical_testing(df, col, col_2='aiContent'):
    """visualize and stats test for dependencies between 
    input: col a numerical feature and col_2, another numerical feature
    output: scatter plot and analysis result
    """
    # analysis by visualization
    # plotting the helpfulScore against aiContent
    with plt.style.context(['science', 'std-colors']):
        sns.scatterplot(data=df, y="aiContent", x=col, hue="ratingScore")
        plt.title('Standardized Helpfulness Count vs '+ ('AI Content Probability' if col_2 == 'aiContent' else f'{col_2}'))
        plt.show()

    # stats testing
    from scipy.stats import spearmanr, kendalltau, pearsonr
    corr, p_values = spearmanr(df[col],
                               df[col_2])
    print(
        f"\nResults of the Spearman test: Correlation is {corr}, with a p-value of {p_values:.05f}.")
    print(f"{col_2} and {col} are",
          "correlated." if p_values < 0.05 else "not correlated.")

    corr, p_values = kendalltau(df[col],
                                df[col_2])
    print(
        f"\nResults of the Kendall Tau test: Correlation is {corr}, with a p-value of {p_values:.05f}.")
    print(f"{col_2} and {col} are",
          "correlated." if p_values < 0.05 else "not correlated.")

    corr, p_values = pearsonr(df[col],
                              df[col_2])
    print(
        f"\nResults of the Pearson test: Correlation is {corr}, with a p-value of {p_values:.05f}.")
    print(f"{col_2} and {col} are",
          "correlated." if p_values < 0.05 else "not correlated.")


def plotbars(dfs, titles, colors):
    """plots bargraphs of the ratings distribution and returns frequency table
    """
    with plt.style.context(['science', 'std-colors']):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        counts_for_testing = []
    
        # Loop through the subplots and create barplots
        for i, ax in enumerate(axs):
            # frequency distribution table of the ratings
            counts = dfs[i].value_counts()
            # collect the data for statistics testing
            counts_for_testing.append(counts)
            ax.bar(counts.index, counts, width=1, label=titles[i], color=colors[i])
            ax.set_xlabel('Rating')
            ax.legend()
    
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.show()
    return counts_for_testing