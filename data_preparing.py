# import WakaTime
# email
# regulr password
from __future__ import annotations

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import requests
from twelvedata import TDClient
import time
import pickle
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, date
from evaluation_and_simulation import where_the_code_runs

"""
amount of data:
recommended at least 5 years worth of that for daily data.
maybe work on with the stocks that have this much info.
"""


"""
Grade system for the models:
score <= 51% -> bad
51% < score <= 53# -> weak
53% < score <= 55% -> solid
55 < score <= 58% -> fucking great
58% < score -> too good, make no sense, look for problems
"""
# todo: for now there are 1370 row per stock, there are 252ish trading days in a year
#  , we have 1370/252 = 5.4 years worth of data -> enough for XGBoost / linear regression,
#  NOT  enough for NN / LSTMs / Transformers.
# todo: see if you worked on a balanced time frame (#up ~ # down)?
# todo: save all in as parquet.
# todo: connect the whole project to gitHub.
# todo: write here a summery of what we did in the cleaning here.
# todo: Get hourly price data.
# todo: Add error handling.
# todo: looks like there is a lot of hard-coding in functions-> fix
"""
This code process the data and makes it ready for training and testing.
"""

"""
predict -> next-day percent return for each stock
"""

# todo: if you change something in the DataFrame before pickling:
# todo: see if it is correct.
# todo: save it in the test pickle, and only if it is good, i can save in the main pickle.

# todo: add more fields.
# todo: we declarer of the var type in the signature of the functions -> easy to read.
# todo: check if I should run on cloud because I have no GPU




if(where_the_code_runs != 1 and where_the_code_runs != 2):
    raise ValueError('Can select only between 1 or 2.')

############### uploading the data from the csv to pandas df ############################################
# this is a file with 40 tech companies - small sample for now
# columns -> ticker = stock symbol, company_name = full company name, group = rough tech subgroup, include_flag = 1 means include in the universe, notes = short reminder about the company
tech_40_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\tech_universe.csv"
if (where_the_code_runs == 2):
    tech_40_path = r"/content/algo_trading_stocks/data/tech_universe.csv"
df_top_40_tech_companies = pd.read_csv(tech_40_path)

# list of the names of all the companies in the df_top_40_tech_companies
top_40_tech_names = df_top_40_tech_companies['ticker'].tolist()

# we want to see all the cols of the pandas df, if you don't, delete
pd.set_option('display.max_columns', None)
#####################################################################


###################################################pickling function###################################################
def pickling_func(data, file_path: str):
    """
    Save a Python object to a pickle file.

    :param data: The Python object to save
    :param file_path: Path to the pickle file (not an existing one)
    """
    # wb - writing in binary mode
    with open(file_path,'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle_data(file_path: str):
    """
    Load a Python object from a pickle file.

    :param file_path: Path to the pickle file
    :return: The restored Python object (with the same data type as it was)
    """
    # rb - read binary file
    with open(file_path, 'rb') as f:
        return pickle.load(f)

################ Twelve data this is a website for API calls for stocks data ############################

"""
we need to call the API calls just once,
after that we just pickle the data in a file and then
just call it
"""

# todo: write the API key in a different place
Twelve_data_API_key = '8b8436f89bc649d1921065d6bfca8c60'

# Initialize client with your API key
td = TDClient(apikey=Twelve_data_API_key)

# todo: this is the API doc: https://twelvedata.com/docs#ws-real-time-price, use it

def company_exchange_and_currency_fetcher(ticker_symbol: str):
    """
    This function returns the Exchange and Currency type
    :param ticker_symbol: the symbol of the stock
    :return: a tuple of the Exchange and Currency type
    """
    url = "https://api.twelvedata.com/symbol_search"
    params = {
        "symbol": ticker_symbol,
        'API_KEY': Twelve_data_API_key
    }

    data = requests.get(url, params = params).json()
    exchange = data['data'][0]['exchange']
    currency = data['data'][0]['currency']

    return exchange, currency

"""
A list of all the acceptable_intervals
"""
acceptable_intervals = ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week', '1month']
def company_earliest_timestamp_fetcher(ticker_symbol: str, interval: str = '1day') -> pd.Timestamp:
    """
    This function returns the Earliest Timestamp of a given stock.
    :param ticker_symbol: The symbol of the stock we want to get its Earliest Timestamp.
    :param interval: The interval we want (Earliest day, Earliest day+hour...)
                     1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    :return: The Earliest Timestamp of a given stock.
    """

    if (interval not in acceptable_intervals):
        raise ValueError('interval must be in the acceptable_intervals list.')

    # This is the API endpoint for earliest_timestamp + symbol + interval + API key
    url = (
        f'https://api.twelvedata.com/earliest_timestamp'
        f'?symbol={ticker_symbol}'
        f'&interval={interval}'
        f'&apikey={Twelve_data_API_key}'
    )

    data = requests.get(url).json()

    if "datetime" not in data:
        raise ValueError(f"API error: {data}")

    earliest_timestamp = pd.to_datetime(data["datetime"])
    return earliest_timestamp

def companies_list_earliest_timestamp_dict_fetcher(symbols_list: list[str], interval: str = '1day')\
        -> dict[str, pd.Timestamp]:
    """
    Loops over a list od symbols and returns a dict with -> key: symbol, value: Earliest Timestamp
    :param symbols_list: The list of the relevant symbols.
    :param interval: The wanted interval.
    :return: A dict with -> key: symbol, value: Earliest Timestamp
    """
    symbols_earliest_timestamp = {}

    if (interval not in acceptable_intervals):
        raise ValueError('interval must be in the acceptable_intervals list.')

    counter = 1
    print(f'start fetching the earliest_timestamps for the {symbols_list} stocks.')
    # we have to add a counter because we are limited to 8 requests per minute.
    for symbol in symbols_list:
        symbols_earliest_timestamp[symbol] = company_earliest_timestamp_fetcher(symbol, interval)
        if counter % 8 == 0 and counter < len(symbols_list):
            print("Reached minute credit limit, sleeping for 60 seconds...")
            time.sleep(60)
        counter += 1

    return symbols_earliest_timestamp

# the path of the Earliest_Timestamps for the top 40 teach companies
pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\Earliest_Timestamps_daily_top_40_teach_companies_data.pkl"
if (where_the_code_runs == 2):
    pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies = r"pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies"

"""
creating a dict of the Earliest_Timestamps_daily of the 40_tech_companies and pickle it.
"""
# Earliest_Timestamps_top_40_tech_companies_daily = companies_list_earliest_timestamp_dict_fetcher(top_40_tech_names) # called it once
# pickling_func(Earliest_Timestamps_top_40_tech_companies_daily, pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies) # called it once, now it is saved
Earliest_Timestamps_top_40_tech_companies_daily = unpickle_data(pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies)

# To excess the timestamp you write -> Earliest_Timestamps_top_40_tech_companies_daily[symbol_name]

"""
Doing the same a dict of the Earliest_Timestamps_hourly of the 40_tech_companies and pickle it.
"""
pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\Earliest_Timestamps_hourly_top_40_teach_companies_data.pkl"
if(where_the_code_runs == 2):
    pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies = r"/content/algo_trading_stocks/data/Earliest_Timestamps_hourly_top_40_teach_companies_data.pkl"


"""
# NOTICE: the code works but we will get diffrent time frames when switching the interval ->
 APPL -> hourly: 2019-01-07 09:00:00, daily: 1980-12-12 00:00:00
"""

# Earliest_Timestamps_top_40_tech_companies_hourly = companies_list_earliest_timestamp_dict_fetcher(top_40_tech_names,'1h') # called it
# pickling_func(Earliest_Timestamps_top_40_tech_companies_hourly, pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies) # called it once, now it is saved
Earliest_Timestamps_top_40_tech_companies_hourly = unpickle_data(pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies)


# todo: Maybe need to fix here as well the part of the start date
def company_data_prices_fetcher(ticker_symbol: str, wanted_interval: str, how_many_intervals: int,
                                start_date:  pd.Timestamp |None = None, end_date: pd.Timestamp |None = None):
    """
    This function returns the Price bars data
    NOTICE: the error handling is builtin the API.
    :param ticker_symbol: the symbol of the stock
    :param wanted_interval: the wanted interval: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    :param how_many_calls: the number of calls.
    :param start_date: The code won't take any rows before this date.
    :param end_date: The code won't take any rows after this date.
    :return: A pandas DF for the company we asked with: symbol, exchange, currency, open, high, low, close, volume
    """
    # check that the end date is bigger than the start date (if they exist).
    if (start_date is not None) and (end_date is not None) and (start_date > end_date):
        raise ValueError("start_date must be <= end_date")

    pands_df = td.time_series(
        symbol=ticker_symbol,
        interval=wanted_interval,
        outputsize=how_many_intervals,
        start_date = start_date,
        end_date = end_date
    ).as_pandas()

    pands_df = pands_df.reset_index()

    # adding the symbol of the company to the pandas df
    pands_df.insert(0, 'symbol', ticker_symbol)

    # adding the exchange and the currency of the stock
    exchange, currency = company_exchange_and_currency_fetcher(ticker_symbol)
    pands_df.insert(1, 'exchange', exchange)
    pands_df.insert(2, 'currency', currency)
    return pands_df



# todo: Think about getting Fundamental data as well


# todo: we need to make sure the functions can handle if there are no trading data in the date we put,
#  start from min_date > 5000 days ago, for each stock
def all_df_creator(ticker_list: list[str], wanted_interval: str, how_many_intervals: int,
                   start_date:  pd.Timestamp |None = None, end_date: pd.Timestamp |None = None,
                   forward_or_backward_flag: int | None = None,
                   check_Earliest_Timestamp_for_df_flag: int | None = None):
    """
    Returns a pandas df with the data of all the companies in company_names_list.
    NOTICE: the error handling is builtin the API.
    :param ticker_list: List of the names of the relevant companies
    :param wanted_interval: Interval -> 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    :param how_many_intervals: How many interval records do we want
            NOTICE: the range of is [1,5000]
            NOTICE: if you want daily prices, hour prices, and min for x days you need to so.
            NOTICE: if you use the function in different days, the time frame will shift as well
    :param start_date: The code won't take any rows before this date.
    :param end_date: The code won't take any rows after this date.
    :param forward_or_backward_flag: A flag to say if we start from today and go back, or start from the
                                     start_date and go forward.
                                     forward_or_backward_flag = None -> from today and go back.
                                     forward_or_backward_flag = 1 ->  start_date and go forward.
    :param check_Earliest_Timestamp_for_df_flag: This flag says if we need to check the Earliest_Timestamp_for_df
                                                 or not, because if we just work on a known df, we don't need to waste
                                                 time and API calls on getting the Earliest_Timestamps, we can get
                                                 it once and pickle it.
                                                 For now the default one is the
                                                 check_Earliest_Timestamp_for_df_flag = None -> no need to check for df.
                                                 check_Earliest_Timestamp_for_df_flag = 1 -> need to check for df.
    :return: A pandas df with the data of all the companies in company_names_list
    """
    dfs = []

    if (wanted_interval not in acceptable_intervals):
        raise ValueError('Wanted_interval must be in the acceptable_intervals list!')

    if (how_many_intervals <= 0 or 5000 < how_many_intervals ):
        raise ValueError('how_many_intervals must be in [1,5000]')

    if not((forward_or_backward_flag is None) or (forward_or_backward_flag == 1)):
        raise ValueError('invalid forward_or_backward_flag value!')

    if not ((check_Earliest_Timestamp_for_df_flag is None) or (check_Earliest_Timestamp_for_df_flag == 1)):
        raise ValueError('invalid check_Earliest_Timestamp_for_df_flag value!')

    companies_list_earliest_timestamp_dict = None
    if(check_Earliest_Timestamp_for_df_flag == 1):
        # a dict of the earliest_timestamp of all the companies in the given df
        companies_list_earliest_timestamp_dict = (companies_list_earliest_timestamp_dict_fetcher
                                                  (ticker_list, wanted_interval))

    # Means we use the default df -> Earliest_Timestamps_top_40_tech_companies_daily or
    # Earliest_Timestamps_top_40_tech_companies_hourly
    # in python we compare strings with '=='
    else:
        if(wanted_interval == '1day'):
            companies_list_earliest_timestamp_dict = Earliest_Timestamps_top_40_tech_companies_daily
        elif(wanted_interval == '1h'):
            companies_list_earliest_timestamp_dict = Earliest_Timestamps_top_40_tech_companies_hourly
        # for now, we only have default df only for these two
        else:
            raise ValueError('There is not a default df for this wanted_interval !')


    # the api allows 8 ticker in a min or less, so we added a counter
    # after 8 companies, we sleep for a min
    for i, ticker in enumerate(ticker_list, start=1):
        # if we are getting data from today backwards, forward_or_backward_flag is None ->
        # there is no need to check what is the Earliest Timestamp.
        # if we reach it we will stop.
        # if we want to start for date x and go forward, forward_or_backward_flag -> we need to now what is the Earliest Timestamp.

        if(ticker not in companies_list_earliest_timestamp_dict):
            raise ValueError('This symbol does not exists in the companies_list_earliest_timestamp_dict!')

        ticker_earliest_timestamp = companies_list_earliest_timestamp_dict[ticker]

        # so we won't override the start_date, end_date for all the symbols
        # each symbol we check separately
        ticker_start_date = start_date
        ticker_end_date = end_date
        if (ticker_start_date is not None) and (ticker_start_date < ticker_earliest_timestamp):
            # adjust the start and end date
            if ticker_end_date is not None:
                delta_between_start_and_end = ticker_end_date - ticker_start_date
                ticker_start_date = ticker_earliest_timestamp
                ticker_end_date = ticker_start_date + delta_between_start_and_end
            else:
                ticker_start_date = ticker_earliest_timestamp

        company_data_df = company_data_prices_fetcher(
            ticker_symbol=ticker,
            wanted_interval=wanted_interval,
            how_many_intervals=how_many_intervals,
            start_date = ticker_start_date,
            end_date = ticker_end_date
        )
        dfs.append(company_data_df)
        # If you hit 8 symbols, wait for the next minute
        if i % 8 == 0 and i < len(ticker_list):
            print("Reached minute credit limit, sleeping for 60 seconds...")
            time.sleep(60)

    return pd.concat(dfs, ignore_index=True)


########################################################################################

###################### pickling the data so we do not have to upload the data each time ################################
###################### we have to create a file to save the data so we can  just exsses it #############################


# five_thousand_days_data = all_df_creator(top_40_tech_names, '1day',5000) # called it once, now it is saved
# getting the stocks data

pickle_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_days_data.pkl"
if(where_the_code_runs == 2):
    pickle_file_path = r"/content/algo_trading_stocks/data/five_thousand_days_data.pkl"
# five_thousand_days_data_pickle = pickling_func(five_thousand_days_data, pickle_file_path) # called it once, now it is saved
five_thousand_days_data_df = unpickle_data(pickle_file_path)



# check code to see how many records per company
# company_row_counts = five_thousand_days_data.groupby('symbol').size().sort_values()
# print(company_row_counts)
# makes sense we didn't get 40*5,000 , not all companies are active for that long -> 160,000 records makes sense

# there is no need to pickle the 40_tech.csv and names list




############################### geting more fundmental data ############################


"""
Calculating and adding columns manually section
"""

def get_percent_change_from_x_closing_days_ago(symbol: str, date: pd.Timestamp, start_days_ago: int = 7,
                                               df: pd.DataFrame| None = five_thousand_days_data_df):
    """
    Return the percent change from the current closing price to the closing price
    found starting X calendar days ago, moving further back until a trading record is found.

    :param symbol: Stock ticker symbol
    :param date: Reference date
    :param start_days_ago: Initial number of days to go back
    :return: Percent change, or np.nan if no valid previous record is found
    """
    days_ago = start_days_ago
    start_date = pd.Timestamp(date).normalize()

    row_of_current_date = df[
        (df["symbol"] == symbol) &
        (df["datetime"].dt.normalize() == start_date)
    ]

    if (start_days_ago <= 0):
        raise ValueError('start_days_ago must be positive.')

    if row_of_current_date.empty:
        return np.nan

    current_date_close_price = row_of_current_date["close"].iloc[0]

    while True:
        date_days_ago = (start_date - pd.Timedelta(days=days_ago)).normalize()

        relevant_row = df[
            (df["symbol"] == symbol) &
            (df["datetime"].dt.normalize() == date_days_ago)
        ]

        if not relevant_row.empty:
            close_price_days_ago = relevant_row["close"].iloc[0]
            return ((current_date_close_price - close_price_days_ago) / close_price_days_ago) * 100

        if days_ago > 20:
            return np.nan

        days_ago += 1


def adding_col_with_values(df: pd.DataFrame, new_col_name: str, function):
    """
    Adds a col with values to the df.
    :param df: the df
    :param new_col_name: the name of the new col.
    :param function: the function we use to calculate the value.
    :return: The df with the new col.
    """
    values = []

    if(new_col_name is None):
        raise ValueError ('new_col_name can not be null')
    if( new_col_name in df.columns.tolist()):
        raise ValueError('The df has a col with this name already')

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        values.append(function(row["symbol"], row["datetime"]))

        if i % 1000 == 0:
            print(f"Processed {i} rows")

    df[new_col_name] = values
    return df

# calling the function and adding the pref_week data
# five_thousand_days_data_df = adding_col_with_values(five_thousand_days_data_df, 'pref_week', get_percent_change_from_x_closing_days_ago)
# print(five_thousand_days_data_df.head(8))

# todo: call the function again with pref_month and write 30 days in the function, not 7
# pickling the new df
# pickling_func(five_thousand_days_data_df, pickle_file_path) # called it once, now it is saved
# five_thousand_days_data_df = unpickle_data(pickle_file_path)
# five_thousand_days_data_df = five_thousand_days_data_df.rename(columns={'datetime': 'date'})
# print(five_thousand_days_data_df.head(8))


######################## adding more fields #################################333
################################ works -> add the pref_week col and pickeld the data ###################################








############ adding more feilds i can compute here ################



# The function is right but really slow, each time filters the df,
# will write the same function but much faster
# def daily_return_percentage_calculator(symbol: str, date: pd.Timestamp):
#     """
#     Calculates the daily return percentage of a stock -> (p(t)-p(t-1))/p(t-1)
#     :param symbol: The symbol of the stock
#     :param date: The date
#     :return: (close_p(date)-close_p(date-1))/close_p(date-1)
#     """
#
#     # finding the close_p of a stock in a date
#     relevant_date = pd.Timestamp(date).normalize()
#     relevant_row = five_thousand_days_data_df[
#         (five_thousand_days_data_df["symbol"] == symbol) &
#         (five_thousand_days_data_df["date"].dt.normalize() == relevant_date)
#     ]
#
#     if relevant_row.empty:
#         return np.nan
#
#     closing_price_relevant_date = relevant_row['close'].iloc[0]
#
#     # doing the same for previous trading date
#
#     days_ago = 1
#     # the previous trading day is not necessarily one day before
#     closing_price_one_trading_day_before = None
#     while closing_price_one_trading_day_before is None and days_ago <= 20:
#         date_one_trading_day_before = (relevant_date - pd.Timedelta(days=days_ago)).normalize()
#         relevant_row_one_trading_day_before = five_thousand_days_data_df[
#             (five_thousand_days_data_df["symbol"] == symbol) &
#             (five_thousand_days_data_df["date"].dt.normalize() == date_one_trading_day_before)
#         ]
#
#         if not relevant_row_one_trading_day_before.empty:
#             closing_price_one_trading_day_before = relevant_row_one_trading_day_before['close'].iloc[0]
#
#         days_ago += 1
#
#     if closing_price_one_trading_day_before is None:
#         return np.nan
#
#     daily_return_percentage = (closing_price_relevant_date - closing_price_one_trading_day_before)/closing_price_one_trading_day_before
#
#     return daily_return_percentage








# good to check if a function is working on this.

# the old function was right but slow
"""
Your old version did this for every row:

search the whole DataFrame for the stock and current date

search the whole DataFrame again for earlier dates

repeat until it finds a match

That is very expensive.

This version:

sorts once

groups once

shifts once

computes the formula in one vectorized operation

So instead of thousands of repeated searches, pandas does one bulk operation.
"""

def add_days_ago_return_percentage_fast(df: pd.DataFrame, days_ago: int, new_col_name: str) -> pd.DataFrame:
    """
    Add daily return percentage per stock using the previous trading row.

    :param df: DataFrame with at least ['symbol', 'date', 'close']
    :param days_ago: Initial number of days to go back
    :return: DataFrame with new new_col_name column
    """

    # coping the df
    df = df.copy()

    # Converts the date column into proper pandas datetime format.
    df["date"] = pd.to_datetime(df["date"])

    # Sorts the rows first by stock symbol, then by date.
    # resets the index of the rows after changing the order of them
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Sorts the rows first by stock symbol, then by date.
    # taking on the 'close' col and 'symbol'
    # shifting the close one row down, so p_close(t+1) is now p(t)
    prev_close_days_ago = df.groupby("symbol")["close"].shift(days_ago)

    # pandas matches values by index. so it know what to dived from what
    """
    before the shift down:
    index	symbol	date	close
    0	AAPL	day1	100
    1	AAPL	day2	103
    
    after the shift down by 1 day:
    index	prev_close
    0	        NaN
    1	        100
    2	        103
    
    computes: df['close'] - pre_close:
    row 0: 100 - NaN
    row 1: 103 - 100
    row 2: 101 - 103
    """
    df[new_col_name] = (df["close"] - prev_close_days_ago) / prev_close_days_ago

    return df


first_row = five_thousand_days_data_df.iloc[0]


def fast_perf_x_trading_days_ago(df: pd.DataFrame, days_ago: int, col_name: str) -> pd.DataFrame:
    """
    Add x_trading_days_ago return percentage per stock using the previous x_trading_days_ago trading row.

    :param df: DataFrame with at least ['symbol', 'date', 'close']
    :return: DataFrame with new 'pref' column
    """

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    close_x_days_ago = df.groupby("symbol")['close'].shift(days_ago)

    df[col_name] = (df['close'] - close_x_days_ago)/close_x_days_ago

    return df


def relative_field_x_days(df: pd.DataFrame, num_of_days: int,new_col_name: str, relevant_col_name: str) -> pd.DataFrame:
    """
    Add relative_field_x_days per stock using the previous relative_volume_x_days trading.
    relative_vol_x = vol_t/ avg_vol_past_x_days

    :param relevant_col_name: The name of the col we will take the info from
    :param df: The df with the data
    :param num_of_days: The number of days to use for the past volume avg calculation
    :param new_col_name: The name of the new column
    :return:
    """

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)



    # to calculate the avd on the pre num_of_days, we'll create a function that runs on the
    # num_of_days amount of previous row, and returns the avg of the volume
    # it would be much smarter to calculate the avg once, and then just update it each time
    # like avg-first_day/20 + new_day/20
    # we will use rolling window for this idea

    rolling_avg = (
        df.groupby('symbol')[relevant_col_name]
        .transform(lambda s: s.shift(1).rolling(window=num_of_days).mean())
    )

    df[new_col_name] = df[relevant_col_name] / rolling_avg
    return df



# adding the now cols and pickling again -> run once
# five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df)
# five_thousand_days_data_df = fast_perf_x_trading_days_ago(five_thousand_days_data_df, 30, 'perf_month')
# five_thousand_days_data_df = relative_field_x_days(five_thousand_days_data_df, 20, "relative_volume_20_days", relevant_col_name="volume") -> checked, it is right


# adding the ret_2, ret_3, ret_5, ret_10
# five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,2,'ret_2')
# five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,3,'ret_3')
# five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,5,'ret_5')
# five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,10,'ret_10')





# the name of the current cols:
# Index(['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low',
#        'close', 'volume', 'pref_week', 'daily_return_percentage', 'perf_month',
#        'relative_volume_20_days'],
#       dtype='object')


# this is the path for experimenting
experiment_pickle_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_days_data_experiment.pkl"
experiment_train_and_validation_pickle_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\experiment_train_and_validation_data.pkl"
experiment_test_pickle_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\experiment_test_data.pkl"
if(where_the_code_runs == 2):
    experiment_pickle_file_path = r"/content/algo_trading_stocks/data/five_thousand_days_data_experiment.pkl"
    experiment_train_and_validation_pickle_file_path = r"/content/algo_trading_stocks/data/experiment_train_and_validation_data.pkl"
experiment_test_pickle_file_path = r"/content/algo_trading_stocks/data/experiment_test_data.pkl"

# one hot encoding but keeping the 'symbol' col
# symbol_dummies = pd.get_dummies(five_thousand_days_data_df['symbol'], prefix='symbol')
# five_thousand_days_data_experiment_df = pd.concat([five_thousand_days_data_df, symbol_dummies], axis=1)

# printing the unique values of the 'exchange' and 'currency'
# print(five_thousand_days_data_df['exchange'].unique()) -> checked and all the stocks are from ['NASDAQ' 'NYSE']
# print(five_thousand_days_data_df['currency'].unique()) -> checked and all the stocks price is in USD

# exchange_dummies = pd.get_dummies(five_thousand_days_data_experiment_df['exchange'], prefix='exchange')
# five_thousand_days_data_experiment_df = pd.concat([five_thousand_days_data_experiment_df, exchange_dummies], axis=1)

# we saved the last version on the pickle_file_path as well, and saved it on the main df name
# pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved
# five_thousand_days_data_experiment_df = unpickle_data(experiment_pickle_file_path) # for now, with symbol one hot encoding




################################# adding more fields to the model #################################


# SMA_20_t = 1/20 * (close_(t) + close_(t-1)+...+close_(t-19))
# SMA_50_t = 1/20 * (close_(t) + close_(t-1)+...+close_(t-19))
# sma20_gap = (close_t/ sma_t - 1) * 100
# sma50_gap = (close_t/ sma_t - 1) * 100


# checked -> calculates correctly
def calculate_rolling_mean_for_x_days (df: pd.DataFrame, field_name: str, num_of_days: int, new_col_name: str, after_col_put_new_col: str = None) -> pd.DataFrame:
    """
    This calculates the AVG values of the field_name of a stock for num_of_days
    and addes it as a col to the df.
    If a row doesn't have 20 days before it, it will put NA in the field.
    if we want partial AVG, write min_periods = 1
    for the amount of data we have per stock, even 50 row to delete is fine. (3.75*365).
    :param df: The df
    :param field_name: The name of the field we calculate the average on
    :param num_of_days: Num of days to calculate the average on
    :param new_col_name: The name of the new column
    :param after_col_put_new_col: The name of the col we want to put our new col after
    :return: The df with the new column and its values
    """

    # copying
    df = df.copy()
    # make sure the format is right
    df['date'] = pd.to_datetime(df['date'])
    # sort by symbol and then date with index reset
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    df[new_col_name] = (
        df.groupby("symbol")[field_name]
        .transform(lambda s: s.rolling(window=num_of_days, min_periods=num_of_days).mean())
    )

    if (after_col_put_new_col is not None):
        new_col = df.pop(new_col_name)
        index_of_new_col = df.columns.get_loc(after_col_put_new_col) + 1
        df.insert(index_of_new_col, new_col_name, new_col)

    return df

# called once and pickled
# five_thousand_days_data_experiment_df = calculate_rolling_mean_for_x_days(five_thousand_days_data_experiment_df, 'close', 20, 'SMA_20', 'ret_10')
# five_thousand_days_data_experiment_df = calculate_rolling_mean_for_x_days(five_thousand_days_data_experiment_df, 'close', 50, 'SMA_50', 'SMA_20')
# print(five_thousand_days_data_experiment_df.iloc[40:60])
# print(five_thousand_days_data_experiment_df.columns.get_loc('ret_10'))

# pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved
# five_thousand_days_data_experiment_df = unpickle_data(experiment_pickle_file_path)


def add_sma_x_gap_percent(df: pd.DataFrame, sma_col_name: str, new_col_name: str, after_col_put_new_col: str = None) -> pd.DataFrame:
    """
    Add the sma_x_gap col to the df
    The formoula is  ((close- SMA_x)/SMA_x) * 100
    :param df: The df
    :param sma_col_name: The name of relevant col
    :param new_col_name: The name of the new col
    :param after_col_put_new_col: The name of the col we want to put our new col after
    :return:
    """
    df = df.copy()

    df[new_col_name] = ((df['close'] - df[sma_col_name])/df[sma_col_name]) * 100

    if (after_col_put_new_col is not None):
        new_col = df.pop(new_col_name)
        index_of_new_col = df.columns.get_loc(after_col_put_new_col) + 1
        df.insert(index_of_new_col, new_col_name, new_col)

    return df

# five_thousand_days_data_experiment_df = add_sma_x_gap_percent(five_thousand_days_data_experiment_df, 'SMA_20', 'SMA_20_gap_percent', 'SMA_50')
# five_thousand_days_data_experiment_df = add_sma_x_gap_percent(five_thousand_days_data_experiment_df, 'SMA_50', 'SMA_50_gap_percent', 'SMA_20_gap_percent')
# pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved
# five_thousand_days_data_experiment_df = unpickle_data(experiment_pickle_file_path)
#
#
# five_thousand_days_data_experiment_df = five_thousand_days_data_experiment_df.copy()
# five_thousand_days_data_experiment_df["date"] = pd.to_datetime(five_thousand_days_data_experiment_df["date"])
# five_thousand_days_data_experiment_df = (
#     five_thousand_days_data_experiment_df
#     .sort_values(["symbol", "date"])
#     .reset_index(drop=True)
# )
#
# # same-day return: keep as feature
# five_thousand_days_data_experiment_df["daily_return_percentage"] = (
#     five_thousand_days_data_experiment_df.groupby("symbol")["close"].pct_change(1)
# )
#
# # true next-day target -> this is a ratio not percentage
# five_thousand_days_data_experiment_df["next_day_return"] = (
#     five_thousand_days_data_experiment_df.groupby("symbol")["close"].shift(-1)
#     / five_thousand_days_data_experiment_df["close"] - 1
# )

# used once when needed -> pickle -> delete lines of call
def move_col_position_in_df(df: pd.DataFrame, name_of_col_to_move: str,
                            index_of_new_position: int | None = None,
                            name_of_col_to_move_after: str | None = None) -> pd.DataFrame:
    """
    Moved a col in df to a different position, based on name_of_col_to_move OR index.
    :param df: The df we change the location of a col in.
    :param name_of_col_to_move: The name of the col we move.
    :param index_of_new_position: The index we want to move the col to.
    :param name_of_col_to_move_after: The name of the col we want to move
                                      our col to be right after it.
    :return: The change df.
    """
    col_to_mov = df.pop(name_of_col_to_move)

    if ( ((name_of_col_to_move_after is not None) and (index_of_new_position is not None)) or
          ((name_of_col_to_move_after is None) and (index_of_new_position is None))):
        raise ValueError('exctly one of name_of_col_to_move_after, index_of_new_position should not be None')

    if (index_of_new_position is not None):
        df.insert(index_of_new_position, name_of_col_to_move, col_to_mov)

    if(name_of_col_to_move_after is not None):
        index_of_new_col = df.columns.get_loc(name_of_col_to_move_after) + 1
        df.insert(index_of_new_col, name_of_col_to_move, col_to_mov)

    return df


def add_col_intraday_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a intraday_range col to the df.
    if called again, overwrites old values.
    intraday_range =  ((high_t - low_t) / close_t(stock)) * 100
    high_t, low_t, close_t are data point of A STOCK on day t
    for now there is only industry, if we add more, will need
    to do a one per industry cause their volatility is different.
    :param df: The df.
    :return: The df with the intraday_range col.
    """
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    df['intraday_range_percent'] = ( (df['high'] - df['low']) / df['close']) * 100

    return df









################################### spliting the data to train_val_set and test_set###################################

# five_thousand_days_data_experiment_df = unpickle_data(experiment_pickle_file_path)
#
#
# print(five_thousand_days_data_experiment_df.head(3))
#
# # from the test i checked, it fucked up the XGBoost model, so drop it for now
# five_thousand_days_data_experiment_df = five_thousand_days_data_experiment_df.drop(columns=['intraday_range_percent'])



"""
data cleaning section
"""

def keep_common_dates_only(df: pd.DataFrame,
                           symbol_col: str = 'symbol',
                           date_col: str = 'date') -> pd.DataFrame:
    """
    Filters the DataFrame to keep only the dates that exist for every symbol.
    """
    # 1. Get the list of unique symbols
    symbols = df[symbol_col].unique()

    if len(symbols) <= 1:
        return df  # Nothing to align if there's only one symbol

    # 2. Find the intersection of dates across all symbols
    # We create a list of sets (one set of dates per symbol)
    date_sets = df.groupby(symbol_col)[date_col].apply(set)

    # Use set.intersection with unpacking (*) to find dates common to all
    common_dates = set.intersection(*date_sets)

    # 3. Filter the original dataframe
    df_filtered = df[df[date_col].isin(common_dates)].copy()

    # --- CHECKS ---
    original_dates_count = df[date_col].nunique()
    final_dates_count = len(common_dates)

    print(f"Alignment Complete:")
    print(f"- Original unique dates: {original_dates_count}")
    print(f"- Common dates kept: {final_dates_count}")
    print(f"- Dates dropped: {original_dates_count - final_dates_count}")

    if final_dates_count == 0:
        print("Warning: No common dates found! The resulting DataFrame is empty.")

    return df_filtered.sort_values([date_col, symbol_col])

def create_df_of_x_percent_of_the_rows(df: pd.DataFrame, percent_to_take: float | int, from_end_or_start: str = 'start') -> pd.DataFrame:
    """
    Return a df of just the last/first percent_to_take%.
    :param df: The df.
    :param percent_to_take: the % of rows to take. ( percent_to_take in [0,100])
    :param from_end_or_start: a flag that says from where to start taking the rows.
    :return: A df of just the last/first percent_to_take%.
    """
    if not 0 <= percent_to_take <= 100:
        raise ValueError('percent must be between 0 to 100')

    number_of_rows_in_df = len(df)

    # math.ceil ensures that 5% of 10 rows returns 1 row instead of 0
    # can also do with return 0 -> it is more common to return 1 row when num_of_row * percent < 1
    number_of_row_to_take = math.ceil(number_of_rows_in_df * (percent_to_take / 100))

    # use of tail or head based on the flag
    # selecting the function dynamically
    method_dict = {
        'start': df.head, # from the start
        'end': df.tail # from the end
    }

    if from_end_or_start not in method_dict:
        raise ValueError('from_end_or_start_flag must be start or end, the default value is start.')
    else:
        df = method_dict[from_end_or_start](number_of_row_to_take)

    return df



def split_df_to_train_val_test(df: pd.DataFrame,
                               percent_for_train: float | int,
                               percent_for_val: float | int,
                               percent_for_test: float | int | None = None,
                               from_end_or_start: str = 'start'
                               )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # todo: check for edge cases, if the split is not even, if it works and takes all the rows
    """
    Splits the df to 3 separate sets, The intersection of the separate is empty.
    train_set, validation_set, test_set.
    We want the to split the df BY DATES NOT BY SYMBOLS, so we have to make sure of 2 thing about the df:
    1. All the symbols have the same dates. -> we already made sure of this
    2. The df is sorted by date from smallest to biggest.
    :param df: The og df.
    :param percent_for_train: percent of the rows to take for the train set (0,100)
    :param percent_for_val: percent of the rows to take for the val set (0,100)
    :param percent_for_test: percent of the rows to take for the test set (0,100)
    :param from_end_or_start: percent of the rows to take for the _ set
    :return: train_set, validation_set, test_set.
    """
    # take the rest to be the test set.
    # all sets must exist, so no 0 or 100
    # todo: add check all symbols have the same date -> maybe as a different function

    # sort by date from smallest to largest
    df = df.sort_values(by='date',ascending=True)
    if (percent_for_test is None):
        if not ( (0 < percent_for_train < 100) and (0 < percent_for_val < 100) and
                 (percent_for_train + percent_for_val) < 100):
            raise ValueError('percent_for_train, percent_for_val, and percent_for_test must be between 0 to 100,'
                             'and sum to 100')
        percent_for_test = 100 - percent_for_train - percent_for_val

    else:
        if not ( (0 < percent_for_train < 100) and (0 <= percent_for_val < 100) and (0 < percent_for_test < 100)
                    and (math.isclose(percent_for_train + percent_for_val + percent_for_test, 100))):
            raise ValueError('percent_for_train, percent_for_val, and percent_for_test must be between 0 to 100, '
                             'and sum to 100')

    length_df = len(df)

    number_of_row_train = int(length_df * (percent_for_train / 100))
    number_of_row_val = int(length_df * (percent_for_val / 100))
    number_of_row_test = length_df - number_of_row_val - number_of_row_train

    if from_end_or_start == 'start':
        train_set = df.iloc[:number_of_row_train]
        validation_set = df.iloc[number_of_row_train:number_of_row_train + number_of_row_val]
        test_set = df.iloc[number_of_row_train + number_of_row_val:]

    elif from_end_or_start == 'end':
        # reverse the calls, cause now we start from the test
        train_set = df.iloc[:number_of_row_test]
        validation_set = df.iloc[number_of_row_test:number_of_row_test + number_of_row_val]
        test_set = df.iloc[number_of_row_test + number_of_row_val:]

    else:
        raise ValueError('from_end_or_start must be start or end.')

    # --- YOUR CHECKS ---

    # 1. Check if we use all rows
    all_rows_used = length_df == (len(train_set) + len(validation_set) + len(test_set))

    # 2. Check for matching rows (Intersection of indices must be empty)
    # Using .intersection() is better than pd.merge because merge checks data values,
    # while intersection checks if it's actually the same row.
    match_train_val = train_set.index.intersection(validation_set.index).empty
    match_val_test = validation_set.index.intersection(test_set.index).empty
    match_train_test = train_set.index.intersection(test_set.index).empty
    no_overlapping_rows = match_train_val and match_val_test and match_train_test

    print(f"All rows accounted for: {all_rows_used}")
    print(f"No overlapping rows: {no_overlapping_rows}\n")

    return train_set, validation_set, test_set



# it is important we work with the same start and end date for all the stocks
# end date -> same for all, the last date when I call to the API
# start date -> can vary, so start from the min-max date (# 2020-09-30 00:00:00)
def same_start_date_for_all_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the df with the same starting date
    for all stocks.
    :param df: The df.
    :return: The df with the same starting date
             for all stocks.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    min_date_all_symbols_have = df.groupby('symbol')['date'].min().max()
    df = df[df['date'] >= min_date_all_symbols_have]

    return df




# here we made sure all the stocks have the same starting date
# five_thousand_days_data_experiment_df = same_start_date_for_all_stocks(five_thousand_days_data_experiment_df)

# we make sure all symbols have the same dates

# five_thousand_days_data_experiment_df = keep_common_dates_only(five_thousand_days_data_experiment_df)
# pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved
five_thousand_days_data_experiment_df = unpickle_data(experiment_pickle_file_path)

# we will take 85% of the data to be in the train_validation set,
# and the other 15% we will take for the test set -> so we won't train our model on the test set.
# train_set, validation_set, data_experiment_test_df = split_df_to_train_val_test(df = five_thousand_days_data_experiment_df,
#                                                                  percent_for_train = 70,
#                                                                  percent_for_val = 15,
#                                                                  percent_for_test = 15,
#                                                                  from_end_or_start = 'start')
#
#
# data_experiment_train_and_validation_df = pd.concat([train_set, validation_set], ignore_index=True)
#
# # it works
# print(f'data_experiment_train_and_validation_df: {len(data_experiment_train_and_validation_df)}')
# print(f'data_experiment_test_df: {len(data_experiment_test_df)}')
# print(f'five_thousand_days_data_experiment_df: {len(five_thousand_days_data_experiment_df)}\n')


# data_experiment_train_and_validation_df = keep_common_dates_only(data_experiment_train_and_validation_df)
# pickling_func(data_experiment_train_and_validation_df, experiment_train_and_validation_pickle_file_path) # called it once, now it is saved
data_experiment_train_and_validation_df = unpickle_data(experiment_train_and_validation_pickle_file_path)

print(f'list of the columns of data_experiment_train_and_validation_df: {data_experiment_train_and_validation_df.columns.tolist()}')

# data_experiment_test_df = keep_common_dates_only(data_experiment_test_df)
# pickling_func(data_experiment_test_df, experiment_test_pickle_file_path) # called it once, now it is saved
data_experiment_test_df = unpickle_data(experiment_test_pickle_file_path)

###################################################checked -> works###################################################
# print(data_experiment_test_df.columns.tolist())
# # this prints the amount of rows each stock has
# print(data_experiment_test_df.groupby('symbol').size().sort_values())
# print(data_experiment_test_df['symbol'].nunique())
# print(data_experiment_train_and_validation_df['symbol'].nunique())
# print(data_experiment_train_and_validation_df.groupby('symbol').size().sort_values())
###################################################checked -> works###################################################

# print the min max (biggest min - the min date that all symbols have)
min_date_all_symbols_have = data_experiment_train_and_validation_df.groupby('symbol')['date'].min().max() # 2020-09-30 00:00:00
max_date = data_experiment_train_and_validation_df.groupby('symbol')['date'].max().max() # 2025-05-20 00:00:00
print(f"min date that all symbols have: {min_date_all_symbols_have}")
print(f"max date in the train_val_set: {max_date}\n")


# todo: find best hyper parameters
# todo: find if this is the right loss function -> rmse is depended on the price scale,

# well leave the last 15% of the dates to be the test set
# every place we run this, we run only of the train and validation sets, no test set.
def data_split_to_train_and_validation(
    df: pd.DataFrame,
    start_date_train_window: date,
    validation_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data two a train and validation set, based on the date the function is getting.
    Train set: rows from start_date_train_window up to but not including validation_date.
    Validate set: rows whose date is exactly validation_date.
    :param df: The df of ONLY THE TRAIN AND VALIDATION SET.
    :param start_date_train_window: The start of the train set.
    :param validation_date: The date of validation -> first trading date after the last train date.
    :return: train_set, validation_set.
    """


    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # check the dates we wrote are in the df
    df_min_date = df['date'].min()

    if (start_date_train_window < df_min_date):
        raise ValueError(f'start_date_train_window: {start_date_train_window} < df_min_date {df_min_date}')

    start_date_train_window = pd.to_datetime(start_date_train_window)
    validation_date = pd.to_datetime(validation_date)

    df = df.sort_values(by="date").reset_index(drop=True)

    # we will split the data to train, validation, test
    train_mask = (df["date"] >= start_date_train_window) & (df["date"] < validation_date)
    train_set = df.loc[train_mask].copy()

    validation_mask = (df["date"] == validation_date)
    validation_set = df.loc[validation_mask].copy()

    return train_set, validation_set

#####################################################scale the data#####################################################
"""
I will min-max scale the X-FEATURES.
It should improve the linear regression and NN models, won't hurt the XGBoost model.
doesn't affect the evaluation_and_simulation logic
"""
# todo: I think i should split the data to train_validation, test here.
#  This means i need to change some of the function:
#  data_prep, run_model_simulation_backvalidation, model_Expending_window_eval add more, that split the.
#  df in them.
#  I than should min-max scale on the x_cols.
#  Than i should min-max scale here.







# delete
print('data_experiment_test_df: ')
print(data_experiment_test_df.columns.tolist())

print('\ndata_experiment_train_and_validation_df: ')
print(data_experiment_train_and_validation_df.columns.tolist())
# delete















