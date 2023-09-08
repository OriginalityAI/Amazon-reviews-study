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


def get_data_for_analysis():
    """
    gets the processed data
    updates column types
    """
    df = pd.read_csv('Data/Data_for_analysis_full.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['ratingScore'] = df['ratingScore'].astype('int').reset_index(drop=True)
    return df


def get_raw_data():
    """returns source data with sensitive information removed"""
    import pandas as pd
    return pd.read_csv('Data/raw_desensitized_data.csv')


def extract_votes(reviewReaction):
    """
    input: reviewReaction (int) e.g. 20 people found this review helpful
    output: the number (int) that finds it helpful or O if it's nan
    """
    if type(reviewReaction) == float:
        return 0
    num = reviewReaction.split()[0].replace(',', '')
    if num == 'One':
        num = 1
    else:
        num = int(num)
    return num


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


def plot_trend(df, start=1990):
    """
    turns the ai_content into a timeseries and plots it
    from the given start date
    """
    data = df.copy()
    data.rename(columns={'ai_content': 'aiContent'})
#     data[data['ai_content'].map(type) !=float] check for errors
    tmp_ = data[['aiContent', 'date']].groupby(
        pd.Grouper(key="date", freq='1D')).mean()
    tmp_ = tmp_.dropna()['aiContent'][str(start):]

    
    tmp = data[['aiContent', 'date']].groupby(
        pd.Grouper(key="date", freq='1M')).mean()
    tmp = tmp.dropna()['aiContent'][str(start):]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = tmp.plot(kind = 'bar', ax=ax, label='aiContent')
    for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), 
                 textcoords='offset points', ha='center', va='bottom')

    
    date_labels = tmp.index.strftime('%Y-%m')
    ax.set_xticklabels(date_labels)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.title('Monthly Average')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    plt.minorticks_on()  # Turn on minor ticks on the axes
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5)) 

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
    import seaborn as sns
    from scipy.stats import f_oneway

    # bar plot
    sns.barplot(data=df, x=col, y=col_2)
    plt.xticks(rotation=90)
    plt.yticks([])
    plt.show()

    # f-oneway stats test
    catgroup = df.groupby(col)[col_2].apply(list)
    p = f_oneway(*catgroup)[1]
    print(f"p value is {p:05f}.")
    print(f"{col_2} and {col} are",
          "correlated." if p < 0.05 else "not correlated.")


def numerical_testing(df, col, col_2='aiContent'):
    # analysis by visualization
    # plotting the helpfulScore against aiContent
    sns.scatterplot(data=df, y="aiContent", x=col, hue="ratingScore")
    plt.title('Standardized Helpfulness Count vs '+ ('AI Content Probability' if col_2 == 'aiContent' else f'{col_2}'))

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


def analyze_timeseries_with_prophet(time_df):
    """
    input: dataframe as a timeseries
    analyzes the timeseries with Prophet forecasting tool
    to study the trend and seasonality
    and plots the series and a trend plot
    can also be used for predictions by changing the
    period parameter (days) in the future declaration
    """
    # required to rename columns to run through the Prophet API
    prophet_df = time_df.reset_index().rename(
        columns={'index': 'ds', 'aiContent': 'y'})

    # instantiate a Prophet object and call fit
    m = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.03
    )
    m.add_seasonality(name='monthly', period=30.5, fourier_order=10)
    m.changepoint = pd.to_datetime(['2022-11-30'])
    forecast = m.fit(prophet_df)
    # make predictions
    future = m.make_future_dataframe(periods=30)  # 1 month
    forecast = m.predict(future)

    # plot predictions
    fig1 = m.plot(forecast)
    plt.show()
    # plot trend
    plt.figure(figsize=(9, 4))
    plt.plot(forecast['ds'], forecast['trend'], label='Trend')
    plt.xlabel('Date')
    plt.ylabel('Average of Reviews with 50% or more AI Content')
    plt.title('Trend Analysis')



    plt.legend()
    plt.grid()

    # add November 30, 2022 vertical marker
    nov302022 = pd.to_datetime('2022-11-30')
    
    aiContent_nov302022 = forecast[forecast['ds'] == nov302022]['trend'].values[0]
    plt.axvline(x=nov302022, color='red', linestyle='--', label='Chat GPT launch')
    plt.text(nov302022, 
             aiContent_nov302022, 
             f'{aiContent_nov302022:.3f} \n@ Nov 30, 2022', 
             rotation=0, 
             va='center', 
             ha='right', 
             bbox=dict(boxstyle='round', alpha=0.1, lw=0, pad=0))
    plt.legend()
    plt.show()

    forecast_filtered = forecast[forecast['ds'] >= nov302022].copy()
    plt.plot(forecast_filtered['ds'], forecast_filtered['trend'], label='Trend', color='g')
    plt.xlabel('Date')
    plt.ylabel('Trend Value')
    plt.title('AI Content Trend Analysis after Chat GPT Launch')
    plt.xticks(rotation=90)
    plt.show()

    
    return m, forecast
