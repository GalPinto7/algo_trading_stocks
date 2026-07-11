# import WakaTime
# email
# regulr password
from __future__ import annotations
from runtime_config import get_where_the_code_runs
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, date


"""
every function with a "#tested" means we tested the function in the test_data_preparing.py and it should be fine.
"""

"""
import the API function.
"""
from twelve_data_api import (
    Twelve_data_API_key,
    td,
    company_exchange_and_currency_fetcher,
    company_earliest_timestamp_fetcher,
    companies_list_earliest_timestamp_dict_fetcher,
    company_data_prices_fetcher,
    all_df_creator,
    acceptable_intervals
)

# todo: Add more error handling.
# todo: check next_day_return, data leak? -> NO!!! we do not train on this, this is only the y_col to test the model.
# todo: delete all the callables that are outside the main function -> DONE
# todo: understand what to comment and what not in the main()
# todo: change the names of these cols [pref_week,daily_return_percentage,perf_month] -> [ret_7,ret_1,ret_30]
"""
This code is responsible of preparing the data.
Cleaning it, and calculating new fields.
"""

"""
Current fields in the csv:
['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low', 'close', 'volume', 'pref_week', 'daily_return_percentage',
 'perf_month', 'relative_volume_20_days', 'ret_2', 'ret_3', 'ret_5', 'ret_10', 'SMA_20', 'SMA_50', 'SMA_20_gap_percent',
  'SMA_50_gap_percent', 'exchange_NYSE', 'exchange_NASDAQ', 'next_day_return',
   'symbol_AAPL', 'symbol_ACN', 'symbol_ADBE', 'symbol_ADP', 'symbol_AMAT',
    'symbol_AMD', 'symbol_ANET', 'symbol_APH', 'symbol_AVGO', 'symbol_CDNS',
     'symbol_CRM', 'symbol_CRWD', 'symbol_CSCO', 'symbol_DDOG', 'symbol_DELL',
      'symbol_DOCU', 'symbol_FTNT', 'symbol_HPE', 'symbol_HPQ', 'symbol_IBM',
       'symbol_INTC', 'symbol_INTU', 'symbol_KLAC', 'symbol_LRCX', 'symbol_MSFT', 'symbol_MU',
        'symbol_NET', 'symbol_NOW', 'symbol_NVDA', 'symbol_ORCL', 'symbol_PANW', 'symbol_PLTR',
         'symbol_QCOM', 'symbol_ROP', 'symbol_SHOP', 'symbol_SNPS', 'symbol_TTD', 'symbol_TXN',
          'symbol_UBER', 'symbol_ZM', 'symbol_AAPL', 'symbol_ACN', 'symbol_ADBE', 'symbol_ADP',
           'symbol_AMAT', 'symbol_AMD', 'symbol_ANET', 'symbol_APH', 'symbol_AVGO', 'symbol_CDNS', 'symbol_CRM',
            'symbol_CRWD', 'symbol_CSCO', 'symbol_DDOG', 'symbol_DELL', 'symbol_DOCU', 'symbol_FTNT', 'symbol_HPE',
             'symbol_HPQ', 'symbol_IBM', 'symbol_INTC', 'symbol_INTU', 'symbol_KLAC', 'symbol_LRCX', 'symbol_MSFT',
              'symbol_MU', 'symbol_NET', 'symbol_NOW', 'symbol_NVDA', 'symbol_ORCL', 'symbol_PANW', 'symbol_PLTR',
               'symbol_QCOM', 'symbol_ROP', 'symbol_SHOP', 'symbol_SNPS',
                'symbol_TTD', 'symbol_TXN', 'symbol_UBER', 'symbol_ZM']
"""

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
# todo: connect the whole project to gitHub. -> DONE.
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



#####################################################Paths of files#####################################################
# tech universe csv
tech_40_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\tech_universe.csv"
tech_40_path_google_colab = r"/content/algo_trading_stocks/data/tech_universe.csv"

# top 40 tech names pickle
pickle_file_path_top_40_tech_names_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\top_40_teach_companies_names.pkl"
pickle_file_path_top_40_tech_names_google_colab = r"/content/algo_trading_stocks/data/top_40_teach_companies_names.pkl"

# Earliest_Timestamps daily pickle
pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\Earliest_Timestamps_daily_top_40_teach_companies_data.pkl"
pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies_google_colab = r"/content/algo_trading_stocks/data/Earliest_Timestamps_daily_top_40_teach_companies_data.pkl"

# Earliest_Timestamps hourly pickle
pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\Earliest_Timestamps_hourly_top_40_teach_companies_data.pkl"
pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies_google_colab = r"/content/algo_trading_stocks/data/Earliest_Timestamps_hourly_top_40_teach_companies_data.pkl"

# five thousand daily prices pickle
pickle_five_thousand_days_data_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_days_data.pkl"
pickle_five_thousand_days_data_file_path_google_colab = r"/content/algo_trading_stocks/data/five_thousand_days_data.pkl"

# experiment full data pickle
experiment_pickle_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_days_data_experiment.pkl"
experiment_pickle_file_path_google_colab = r"/content/algo_trading_stocks/data/five_thousand_days_data_experiment.pkl"

# experiment train and validation pickle
experiment_train_and_validation_pickle_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\experiment_train_and_validation_data.pkl"
experiment_train_and_validation_pickle_file_path_google_colab = r"/content/algo_trading_stocks/data/experiment_train_and_validation_data.pkl"


# experiment test pickle
experiment_test_pickle_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\experiment_test_data.pkl"
experiment_test_pickle_file_path_google_colab = r"/content/algo_trading_stocks/data/experiment_test_data.pkl"

# just for testing
pickle_five_thousand_days_data_file_path_for_testing_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_days_data_testing.pkl"
pickle_five_thousand_days_data_file_path_for_testing_google_colab = r"/content/algo_trading_stocks/data/five_thousand_days_data_testing.pkl"

########################################################################################################################

############empty vars we will update later - def here so code will not crush in diffrent files how use them############
# Safe defaults.
# They prevent heavy file loading during import.
df_top_40_tech_companies = None
top_40_tech_names = None
Earliest_Timestamps_top_40_tech_companies_daily = None
Earliest_Timestamps_top_40_tech_companies_hourly = None
five_thousand_days_data_df = None
five_thousand_days_data_experiment_df = None
data_experiment_train_and_validation_df = None
data_experiment_test_df = None
min_date_all_symbols_have = None
max_date = None
########################################################################################################################


def Access_the_file_path(where_the_code_runs: int, path_local: str, path_google_colab): # tested
    """
    Returns the relevant file path.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :param path_local: The path of the file in the local computer.
    :param path_google_colab: The path of the file in google colab.
    :return: The path depending on where the code runs.
    """
    if (where_the_code_runs == 1):
        return path_local
    elif (where_the_code_runs == 2):
        return path_google_colab
    else:
        raise ValueError('Can select only between 1 or 2.')

    ############### uploading the data from the csv to pandas df ############################################
# this is a file with 40 tech companies - small sample for now
# columns -> ticker = stock symbol, company_name = full company name, group = rough tech subgroup, include_flag = 1 means include in the universe, notes = short reminder about the company


# we want to see all the cols of the pandas df, if you don't, delete
pd.set_option('display.max_columns', None)
#####################################################################


###################################################pickling function###################################################
def pickling_func(data, file_path: str): #TESTED
    """
    Save a Python object to a pickle file.
    :param data: The Python object to save
    :param file_path: Path to the pickle file (not an existing one)
    """
    if(data is None):
        raise ValueError("data can't be None!")
    elif(file_path is None):
        raise ValueError("file_path can't be None!")
    # wb - writing in binary mode
    with open(file_path,'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle_data(file_path: str): #TESTED
    """
    Load a Python object from a pickle file.
    :param file_path: Path to the pickle file
    :return: The restored Python object (with the same data type as it was)
    """
    # rb - read binary file
    if file_path is None:
        raise ValueError("file_path can't be None!")
    with open(file_path, 'rb') as f:
        return pickle.load(f)



##################################################loaders of the data##################################################
def load_top_40_tech_companies_csv(where_the_code_runs: int) -> pd.DataFrame: #TESTED
    """
    Returns the csv of the top 40 companies, as a pandas dataframe.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: The csv of the top 40 companies, as a pandas dataframe.
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    tech_40_path = Access_the_file_path(where_the_code_runs, tech_40_path_local, tech_40_path_google_colab)
    df_top_40_tech_companies = pd.read_csv(tech_40_path)
    return df_top_40_tech_companies


def load_top_40_tech_companies_names(where_the_code_runs: int) -> list[str]: #TESTED
    """
    Goes to the path of the pkl file with the names of the companies.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: A list of top 40 companies names.
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_top_40_tech_companies_names_pkl = Access_the_file_path(
        where_the_code_runs = where_the_code_runs,
        path_local = pickle_file_path_top_40_tech_names_local,
        path_google_colab=pickle_file_path_top_40_tech_names_google_colab
    )
    return unpickle_data(path_top_40_tech_companies_names_pkl)

def load_Earliest_Timestamps_top_40_tech_companies_daily(where_the_code_runs) -> dict[str, pd.Timestamp]: #TESTED
    """
    Loads the Earliest_Timestamps daily.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: Dict of the Earliest_Timestamps_daily of the 40_tech_companies.
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')
    path_Earliest_Timestamps_top_40_tech_companies_daily_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies_local,
        path_google_colab=pickle_file_path_Earliest_Timestamps_daily_top_40_teach_companies_google_colab
    )
    return unpickle_data(path_Earliest_Timestamps_top_40_tech_companies_daily_pkl)


################ Twelve data this is a website for API calls for stocks data ############################

"""
we need to call the API calls just once,
after that we just pickle the data in a file and then
just call it.
"""

# To excess the timestamp you write -> Earliest_Timestamps_top_40_tech_companies_daily[symbol_name]

"""
Doing the same a dict of the Earliest_Timestamps_hourly of the 40_tech_companies and pickle it.
"""
def load_Earliest_Timestamps_top_40_tech_companies_hourly(where_the_code_runs) -> dict[str, pd.Timestamp]: #TESTED
    """
    Loads the Earliest_Timestamps hourly.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: Dict of the Earliest_Timestamps_hourly of the 40_tech_companies.
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_Earliest_Timestamps_top_40_tech_companies_hourly_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies_local,
        path_google_colab=pickle_file_path_Earliest_Timestamps_hourly_top_40_teach_companies_google_colab
    )
    return unpickle_data(path_Earliest_Timestamps_top_40_tech_companies_hourly_pkl)



def load_five_thousand_days_data_df(where_the_code_runs: int) -> pd.DataFrame: #TESTED
    """
    Loads the five_thousand_days_data_df DataFrame.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: five_thousand_days_data_df
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_five_thousand_days_data_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=pickle_five_thousand_days_data_file_path_local,
        path_google_colab=pickle_five_thousand_days_data_file_path_google_colab
    )

    return unpickle_data(path_five_thousand_days_data_pkl)


def load_five_thousand_days_data_experiment_df(where_the_code_runs: int) -> pd.DataFrame: #TESTED
    """
    Loads the five_thousand_days_data_experiment_df DataFrame.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: five_thousand_days_data_experiment_df
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_experiment_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=experiment_pickle_file_path_local,
        path_google_colab=experiment_pickle_file_path_google_colab
    )

    return unpickle_data(path_experiment_pkl)

def load_data_experiment_train_and_validation_df(where_the_code_runs: int) -> pd.DataFrame: #TESTED
    """
    Loads the data_experiment_train_and_validation_df DataFrame.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: data_experiment_train_and_validation_df
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_train_and_validation_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=experiment_train_and_validation_pickle_file_path_local,
        path_google_colab=experiment_train_and_validation_pickle_file_path_google_colab
    )

    return unpickle_data(path_train_and_validation_pkl)


def load_data_experiment_test_df(where_the_code_runs: int) -> pd.DataFrame: #TESTED
    """
    Loads the data_experiment_test_df DataFrame.
    :param where_the_code_runs: 1 -> local, 2 -> google colab.
    :return: data_experiment_test_df
    """
    if(where_the_code_runs not in [1, 2]):
        raise ValueError('Where code must be either 1 or 2.')

    path_test_pkl = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=experiment_test_pickle_file_path_local,
        path_google_colab=experiment_test_pickle_file_path_google_colab
    )

    return unpickle_data(path_test_pkl)
"""
# NOTICE: the code works but we will get different time frames when switching the interval ->
 APPLY -> hourly: 2019-01-07 09:00:00, daily: 1980-12-12 00:00:00
"""



"""
Calculating and adding columns manually section
"""

def get_percent_change_from_x_closing_days_ago(symbol: str, date: pd.Timestamp, start_days_ago: int = 7,
                                               df: pd.DataFrame | None = None): # tested
    """
    Return the percent change from the current closing price to the closing price
    found starting X calendar days ago, moving further back until a trading record is found.

    :param symbol: Stock ticker symbol
    :param date: Reference date
    :param start_days_ago: Initial number of days to go back
    :return: Percent change, or np.nan if no valid previous record is found
    """

    if df is None:
        raise ValueError('df must be provided.')

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    days_ago = start_days_ago
    start_date = pd.Timestamp(date).normalize()

    row_of_current_date = df[
        (df["symbol"] == symbol) &
        (df["date"].dt.normalize() == start_date)
    ]

    if (start_days_ago <= 0):
        raise ValueError('start_days_ago must be positive.')

    elif row_of_current_date.empty:
        return np.nan

    current_date_close_price = row_of_current_date["close"].iloc[0]

    while True:
        date_days_ago = (start_date - pd.Timedelta(days=days_ago)).normalize()

        relevant_row = df[
            (df["symbol"] == symbol) &
            (df["date"].dt.normalize() == date_days_ago)
        ]

        if not relevant_row.empty:
            close_price_days_ago = relevant_row["close"].iloc[0]
            return ((current_date_close_price - close_price_days_ago) / close_price_days_ago) * 100

        elif days_ago > 20:
            return np.nan

        days_ago += 1


def adding_col_with_values(df: pd.DataFrame, new_col_name: str, function) -> pd.DataFrame: #TESTED
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
    elif( new_col_name in df.columns.tolist()):
        raise ValueError('The df has a col with this name already')

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        values.append(function(row["symbol"], row["date"]))

        if i % 1000 == 0:
            print(f"Processed {i} rows")

    df[new_col_name] = values
    return df









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

def add_days_ago_return_percentage_fast(df: pd.DataFrame, days_ago: int, new_col_name: str) -> pd.DataFrame: #TESTED
    """
    Add daily return percentage per stock using the previous trading row.

    :param df: DataFrame with at least ['symbol', 'date', 'close']
    :param days_ago: Initial number of days to go back
    :return: DataFrame with new new_col_name column
    """
    if(df is None):
        raise ValueError('df must be provided.')
    elif(days_ago <= 0):
        raise ValueError('days_ago must be positive.')
    elif(new_col_name is None):
        raise ValueError('new_col_name can not be null')

    # coping the df
    df = df.copy()

    # Converts the date column into proper pandas datetime format.
    df["date"] = pd.to_datetime(df["date"])

    # Sorts the rows first by stock symbol, then by date.
    # resets the index of the rows after changing the order of them
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Sorts the rows first by stock symbol, then by date.
    # taking on the 'close' col and 'symbol'
    # shifting the close days_ago row down, so if days_ago = 1 -> p_close(t+1) is now p(t)
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


def remove_duplicates_based_on_fields(df: pd.DataFrame, cols_to_check: list[str]) -> pd.DataFrame: # TESTED
    """
    Checks all the row in the df for duplicates based only on the values in the cols in the "cols_to_check" list.
    :param df: The df we work on.
    :param cols_to_check: A list of cols we want to compare the rows based on.
    :return: The df with no 2 rows with the same values in ALL of these fields.
    """
    if(df is None):
        raise ValueError("The df can't be None!")
    elif len(cols_to_check) == 0:
        raise ValueError("The cols_to_check list should not be empty")

    cols_in_df = df.columns.tolist()

    for col in cols_to_check:
        if col not in cols_in_df:
            raise ValueError ('All the cols in the "cols_to_check" must be in the df.')

    df = df.drop_duplicates(subset=cols_to_check)
    return df


def fast_perf_x_trading_days_ago(df: pd.DataFrame, days_ago: int, col_name: str) -> pd.DataFrame: #TESTED
    """
    Add x_trading_days_ago return percentage per stock using the previous x_trading_days_ago trading row.
    NOTICE: THIS IS FOR THE WHOLE DF.
    :param df: DataFrame with at least ['symbol', 'date', 'close']
    :param days_ago:
    :param col_name: relevant col, can be in the df or not.
    :return: DataFrame with new 'pref' column
    """
    if(df is None):
        raise ValueError("Df can't be none.")
    elif(days_ago<=0):
        raise ValueError("days ago can't be none.")
    elif(col_name is None):
        raise ValueError("col_name can not be null.")

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    # groups by stock name, so there are different tables per stock
    close_x_days_ago = df.groupby("symbol")['close'].shift(days_ago)
    # return = (current_price - price_x_days_ago)/price_x_days_ago
    df[col_name] = (df['close'] - close_x_days_ago)/close_x_days_ago

    return df



def fast_perf_x_trading_days_ago_for_last_date(
    new_day_df: pd.DataFrame,
    historic_df: pd.DataFrame,
    days_ago: int,
    name_of_col: str,
) -> pd.DataFrame: #TESTED
    """
    Calculates the x-trading-days-ago return for the new rows only,
    while keeping all historic rows.

    :param new_day_df: DataFrame with the new daily rows.
    :param historic_df: Existing full historical DataFrame.
    :param days_ago: The days ago we want to calculate the "ret_n".
    :param name_of_col: Name of the feature column to calculate.
                        It works both if the col is in the df and if not.
    :return: Full DataFrame = old rows + new rows, with name_of_col calculated for the new rows.
    """
    if new_day_df is None:
        raise ValueError("new_day_df can't be None.")

    elif new_day_df.empty:
        raise ValueError("new_day_df is empty")

    elif historic_df is None:
        raise ValueError("historic_df can't be None.")

    elif historic_df.empty:
        raise ValueError("historic_df is empty")

    elif (days_ago <= 0):
        raise ValueError("The list_days_ago must have at least one day.")


    required_cols = ["symbol", "date", "close"]

    for col in required_cols:
        if col not in new_day_df.columns:
            raise ValueError(f"{col} must be in new_day_df")

        elif col not in historic_df.columns:
            raise ValueError(f"{col} must be in historic_df")

    historic_df = historic_df.copy()
    new_day_df = new_day_df.copy()

    historic_df["date"] = pd.to_datetime(historic_df["date"]).dt.normalize()
    new_day_df["date"] = pd.to_datetime(new_day_df["date"]).dt.normalize()

    # Keep full historic df for the final output.
    full_historic_df = historic_df.copy()

    # Take only the last days_ago rows per symbol as context.
    historic_context_df = (
        historic_df
        .sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .tail(days_ago)
    )

    # Mark rows so we know which rows are new after calculation.
    historic_context_df["_is_new_row"] = False
    new_day_df["_is_new_row"] = True

    # Combine only the needed history + new rows.
    calc_df = pd.concat([historic_context_df, new_day_df], ignore_index=True)

    # Drop duplicates before calculation.
    calc_df = (
        calc_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # Calculate the feature.
    calc_df = fast_perf_x_trading_days_ago(
        df=calc_df,
        days_ago=days_ago,
        col_name=name_of_col
    )

    # Keep only the new rows with the calculated feature.
    calculated_new_rows = calc_df[calc_df["_is_new_row"]].copy()
    calculated_new_rows = calculated_new_rows.drop(columns=["_is_new_row"])

    # The old full df did not have this helper column.
    if "_is_new_row" in full_historic_df.columns:
        full_historic_df = full_historic_df.drop(columns=["_is_new_row"])

    # Add the calculated new rows to the full old df.
    whole_df = pd.concat([full_historic_df, calculated_new_rows], ignore_index=True)

    # Drop duplicates from the final full df.
    whole_df = (
        whole_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    return whole_df





def relative_field_x_days(df: pd.DataFrame, num_of_days: int,new_col_name: str, relevant_col_name: str) -> pd.DataFrame:
    """ #TESTED
    Add relative_field_x_days per stock using the previous relative_volume_x_days trading.
    relative_vol_x = vol_t/ avg_vol_past_x_days

    :param relevant_col_name: The name of the col we will take the info from
    :param df: The df with the data
    :param num_of_days: The number of days to use for the past volume avg calculation
    :param new_col_name: The name of the new column
    :return:
    """

    if(df is None):
        raise ValueError("Df can't be None!")
    elif(num_of_days <= 0):
        raise ValueError("num_of_days must be positive!")
    elif(new_col_name is None):
        raise ValueError("new_col_name can't be None!")
    elif(relevant_col_name is None):
        raise ValueError("relevant_col_name can't be None!")
    elif(relevant_col_name not in df.columns):
        raise ValueError("relevant_col_name must be in df")

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

# todo: needs to be checked
def relative_field_x_days_for_last_date(
    new_day_df: pd.DataFrame,
    historic_df: pd.DataFrame,
    num_of_days: int,
    new_col_name: str,
    relevant_col_name: str
) -> pd.DataFrame:
    """
    Calculates relative_field_x_days only for the new rows,
    then adds the calculated new rows to the full historic df.

    Example:
    relative_volume_20_days = volume_t / average(volume of previous 20 trading rows)

    :param new_day_df: DataFrame with only the new daily rows.
    :param historic_df: Existing full historical DataFrame.
    :param num_of_days: Number of previous trading rows to use.
    :param new_col_name: Name of the new calculated column.
    :param relevant_col_name: Column used for the calculation, for example "volume".
    :return: Full DataFrame = historic rows + calculated new rows, without duplicates.
    """

    if new_day_df is None:
        raise ValueError("new_day_df can't be None!")

    if historic_df is None:
        raise ValueError("historic_df can't be None!")

    if new_day_df.empty:
        raise ValueError("new_day_df is empty!")

    if historic_df.empty:
        raise ValueError("historic_df is empty!")

    if num_of_days <= 0:
        raise ValueError("num_of_days must be positive!")

    if new_col_name is None:
        raise ValueError("new_col_name can't be None!")

    if relevant_col_name is None:
        raise ValueError("relevant_col_name can't be None!")

    required_cols = ["symbol", "date", relevant_col_name]

    for col in required_cols:
        if col not in new_day_df.columns:
            raise ValueError(f"{col} must be in new_day_df")

        if col not in historic_df.columns:
            raise ValueError(f"{col} must be in historic_df")

    historic_df = historic_df.copy()
    new_day_df = new_day_df.copy()

    historic_df["date"] = pd.to_datetime(historic_df["date"]).dt.normalize()
    new_day_df["date"] = pd.to_datetime(new_day_df["date"]).dt.normalize()

    # Keep the full old df for the final output.
    full_historic_df = historic_df.copy()

    # For this calculation we need the previous num_of_days rows per symbol.
    historic_context_df = (
        historic_df
        .sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .tail(num_of_days)
    )

    # Mark which rows are new.
    historic_context_df["_is_new_row"] = False
    new_day_df["_is_new_row"] = True

    # Combine only calculation context + new rows.
    calc_df = pd.concat([historic_context_df, new_day_df], ignore_index=True)

    # Remove duplicates before calculation.
    calc_df = (
        calc_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # Calculate the feature.
    calc_df = relative_field_x_days(
        df=calc_df,
        num_of_days=num_of_days,
        new_col_name=new_col_name,
        relevant_col_name=relevant_col_name
    )

    # Keep only the new rows after calculation.
    calculated_new_rows = calc_df[calc_df["_is_new_row"]].copy()
    calculated_new_rows = calculated_new_rows.drop(columns=["_is_new_row"])

    if "_is_new_row" in full_historic_df.columns:
        full_historic_df = full_historic_df.drop(columns=["_is_new_row"])

    # Add calculated new rows to full old df.
    whole_df = pd.concat([full_historic_df, calculated_new_rows], ignore_index=True)

    # Drop duplicates in the final full df.
    whole_df = (
        whole_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    return whole_df





# the name of the current cols:
# Index(['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low',
#        'close', 'volume', 'ret_7', 'ret_1', 'ret_30',
#        'relative_volume_20_days'],
#       dtype='object')





################################# adding more fields to the model #################################


# SMA_20_t = 1/20 * (close_(t) + close_(t-1)+...+close_(t-19))
# SMA_50_t = 1/20 * (close_(t) + close_(t-1)+...+close_(t-19))
# sma20_gap = (close_t/ sma_t - 1) * 100
# sma50_gap = (close_t/ sma_t - 1) * 100


# checked -> calculates correctly
def calculate_rolling_mean_for_x_days (df: pd.DataFrame, field_name: str, num_of_days: int, new_col_name: str, after_col_put_new_col: str = None) -> pd.DataFrame:
    """ #TESTED
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

    if(df is None):
        raise ValueError("Df can't be None!")
    if(df.empty):
        raise ValueError("df can't be None!")
    if(field_name is None):
        raise ValueError("Field name can't be None!")
    if(num_of_days <= 0):
        raise ValueError("num_of_days must be positive!")
    if(new_col_name is None):
        raise ValueError("new_col_name can't be None!")

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

# todo: need to check this function
def calculate_rolling_mean_for_x_days_for_last_date(
    new_day_df: pd.DataFrame,
    historic_df: pd.DataFrame,
    field_name: str,
    num_of_days: int,
    new_col_name: str,
    after_col_put_new_col: str | None = None
) -> pd.DataFrame:
    """
    Calculates rolling mean only for the new rows,
    then adds the calculated new rows to the full historic df.

    Example:
    SMA_20 = average(close_t, close_t-1, ..., close_t-19)

    :param new_day_df: DataFrame with only the new daily rows.
    :param historic_df: Existing full historical DataFrame.
    :param field_name: Column to calculate rolling mean on, for example "close".
    :param num_of_days: Rolling window size.
    :param new_col_name: Name of the new calculated column.
    :param after_col_put_new_col: Optional column name to insert new_col_name after.
    :return: Full DataFrame = historic rows + calculated new rows, without duplicates.
    """

    if new_day_df is None:
        raise ValueError("new_day_df can't be None!")

    if historic_df is None:
        raise ValueError("historic_df can't be None!")

    if new_day_df.empty:
        raise ValueError("new_day_df is empty!")

    if historic_df.empty:
        raise ValueError("historic_df is empty!")

    if field_name is None:
        raise ValueError("field_name can't be None!")

    if num_of_days <= 0:
        raise ValueError("num_of_days must be positive!")

    if new_col_name is None:
        raise ValueError("new_col_name can't be None!")

    required_cols = ["symbol", "date", field_name]

    for col in required_cols:
        if col not in new_day_df.columns:
            raise ValueError(f"{col} must be in new_day_df")

        if col not in historic_df.columns:
            raise ValueError(f"{col} must be in historic_df")

    historic_df = historic_df.copy()
    new_day_df = new_day_df.copy()

    historic_df["date"] = pd.to_datetime(historic_df["date"]).dt.normalize()
    new_day_df["date"] = pd.to_datetime(new_day_df["date"]).dt.normalize()

    # Keep the full old df for the final output.
    full_historic_df = historic_df.copy()

    # For rolling(window=num_of_days), the new row needs previous num_of_days - 1 rows.
    # Taking num_of_days is also fine and safe.
    historic_context_df = (
        historic_df
        .sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .tail(num_of_days)
    )

    # Mark which rows are new.
    historic_context_df["_is_new_row"] = False
    new_day_df["_is_new_row"] = True

    # Combine only calculation context + new rows.
    calc_df = pd.concat([historic_context_df, new_day_df], ignore_index=True)

    # Remove duplicates before calculation.
    calc_df = (
        calc_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # Calculate the feature.
    calc_df = calculate_rolling_mean_for_x_days(
        df=calc_df,
        field_name=field_name,
        num_of_days=num_of_days,
        new_col_name=new_col_name,
        after_col_put_new_col=after_col_put_new_col
    )

    # Keep only the new rows after calculation.
    calculated_new_rows = calc_df[calc_df["_is_new_row"]].copy()
    calculated_new_rows = calculated_new_rows.drop(columns=["_is_new_row"])

    if "_is_new_row" in full_historic_df.columns:
        full_historic_df = full_historic_df.drop(columns=["_is_new_row"])

    # Add calculated new rows to full old df.
    whole_df = pd.concat([full_historic_df, calculated_new_rows], ignore_index=True)

    # Drop duplicates in the final full df.
    whole_df = (
        whole_df
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    return whole_df



def add_sma_x_gap_percent(df: pd.DataFrame, sma_col_name: str, new_col_name: str, after_col_put_new_col: str = None) -> pd.DataFrame:
    """ #TESTED
    Add the sma_x_gap col to the df
    The formoula is  ((close- SMA_x)/SMA_x) * 100
    :param df: The df
    :param sma_col_name: The name of relevant col
    :param new_col_name: The name of the new col
    :param after_col_put_new_col: The name of the col we want to put our new col after
    :return:
    """
    if (df is None):
        raise ValueError("Df can't be None!")
    if (sma_col_name is None):
        raise ValueError("sma_col_name can't be None!")
    if (new_col_name is None):
        raise ValueError("new_col_name can't be None!")

    df = df.copy()

    df[new_col_name] = ((df['close'] - df[sma_col_name])/df[sma_col_name]) * 100

    if (after_col_put_new_col is not None):
        new_col = df.pop(new_col_name)
        index_of_new_col = df.columns.get_loc(after_col_put_new_col) + 1
        df.insert(index_of_new_col, new_col_name, new_col)

    return df


# todo: finish
def calculate_RSI_for_x_days(df: pd.DataFrame, num_of_days: int| None = 14):
    """
    Receives a df and the number of days we want to calculate the RSI for,
    add the col and calculate the RSI.
    RSI = 100-(100/ (1+RS))
    RS = avg_gain_num_of_days/avg_loss_num_of_days
    avg_gain -> from 1/num_of_days * sum(gain_i) (for i in num_of_days)
                so we dvied by the num of days, but sum only over the days we gained.
    avg_loss -> same idea
    :param df: The df we work on.
    :param num_of_days: The number of days -> RSI_num_of_days
    :return: The df with the RSI_num_of_days col.
    """
    if (df is None):
        raise ValueError("Df can't be None!")
    elif(df.empty):
        raise ValueError("Df can't be empty")

    elif(num_of_days <= 0):
        raise ValueError("num_of_days must be positive!")

    name_of_col = f"RSI_{num_of_days}"







# used once when needed -> pickle -> delete lines of call
def move_col_position_in_df(df: pd.DataFrame, name_of_col_to_move: str,
                            index_of_new_position: int | None = None,
                            name_of_col_to_move_after: str | None = None) -> pd.DataFrame: #TESTED
    """
    Moved a col in df to a different position, based on name_of_col_to_move OR index.
    :param df: The df we change the location of a col in.
    :param name_of_col_to_move: The name of the col we move.
    :param index_of_new_position: The index we want to move the col to.
    :param name_of_col_to_move_after: The name of the col we want to move
                                      our col to be right after it.
    :return: The change df.
    """
    if(df is None):
        raise ValueError("Df can't be None!")
    elif(df.empty):
        raise ValueError("Df can't be empty!")

    col_to_mov = df.pop(name_of_col_to_move)

    if ( ((name_of_col_to_move_after is not None) and (index_of_new_position is not None)) or
          ((name_of_col_to_move_after is None) and (index_of_new_position is None))):
        raise ValueError('exactly one of name_of_col_to_move_after, index_of_new_position should not be None')

    if (index_of_new_position is not None):
        df.insert(index_of_new_position, name_of_col_to_move, col_to_mov)

    if(name_of_col_to_move_after is not None):
        index_of_new_col = df.columns.get_loc(name_of_col_to_move_after) + 1
        df.insert(index_of_new_col, name_of_col_to_move, col_to_mov)

    return df


def add_col_intraday_range(df: pd.DataFrame) -> pd.DataFrame: #TESTED
    """
    Adds an intraday_range col to the df.
    if called again, overwrites old values.
    intraday_range =  ((high_t - low_t) / close_t(stock)) * 100
    high_t, low_t, close_t are data point of A STOCK on day t
    for now there is only industry, if we add more, will need
    to do a one per industry cause their volatility is different.
    :param df: The df.
    :return: The df with the intraday_range col.
    """
    if(df is None):
        raise ValueError("df can't be None!")
    if(df.empty):
        raise ValueError("df can't be empty!")

    df = df.copy()

    df['date'] = pd.to_datetime(df['date'])

    df['intraday_range_percent'] = ( (df['high'] - df['low']) / df['close']) * 100

    return df






"""
data cleaning section
"""

def keep_common_dates_only(df: pd.DataFrame,
                           symbol_col: str = 'symbol',
                           date_col: str = 'date') -> pd.DataFrame: # TESTED
    """
    Filters the DataFrame to keep only the dates that exist for every symbol.
    :param df: The df we work on.
    :param symbol_col: A column of the symbols of the ['APPL','INTEL']
    :param date_col: A column of the dates of the records.
    :return: A df with only the records of the dates that are common for all the stocks.
    """
    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")

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
    """ #TESTED
    Return a df of just the last/first percent_to_take%.
    :param df: The df.
    :param percent_to_take: the % of rows to take. ( percent_to_take in [0,100])
    :param from_end_or_start: a flag that says from where to start taking the rows.
    :return: A df of just the last/first percent_to_take%.
    """
    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")
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
                               )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: #TESTED
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
    :return: train_set, validation_set, test_set.
    """
    # take the rest to be the test set.
    # all sets must exist, so no 0 or 100
    # todo: add check all symbols have the same date -> maybe as a different function
    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")

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

    train_set = df.iloc[:number_of_row_train]
    validation_set = df.iloc[number_of_row_train:number_of_row_train + number_of_row_val]
    test_set = df.iloc[number_of_row_train + number_of_row_val:]


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
def same_start_date_for_all_stocks(df: pd.DataFrame) -> pd.DataFrame: # TESTED
    """
    Returns the df with the same starting date
    for all stocks.
    :param df: The df.
    :return: The df with the same starting date
             for all stocks.
    """
    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    min_date_all_symbols_have = df.groupby('symbol')['date'].min().max()
    df = df[df['date'] >= min_date_all_symbols_have]

    return df





# todo: find best hyper parameters
# todo: find if this is the right loss function -> rmse is depended on the price scale,

# well leave the last 15% of the dates to be the test set
# every place we run this, we run only of the train and validation sets, no test set.
def data_split_to_train_and_validation(
    df: pd.DataFrame,
    start_date_train_window: date,
    validation_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]: #TESTED
    """
    Splits the data two a train and validation set, based on the date the function is getting.
    Train set: rows from start_date_train_window up to but not including validation_date.
    Validate set: rows whose date is exactly validation_date.
    :param df: The df of ONLY THE TRAIN AND VALIDATION SET.
    :param start_date_train_window: The start of the train set.
    :param validation_date: The date of validation -> first trading date after the last train date.
    :return: train_set, validation_set.
    """

    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")

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

def one_hot_encoding(df: pd.DataFrame, field_to_one_hot_encode: str, prefix: str | None = None) -> pd.DataFrame: #TESTED
    """
    This function receives a pandas dataframe and returns a one-hot-encoded dataframe
    on the col we wrote.
    :param df: The df to one hot-encode.
    :param field_to_one_hot_encode: The field we wrote to one-hot-encode.
    :param prefix: The string we want to add to the name of the column we wrote in the new cols.
    :return: The df after the one-hot-encode.
    """

    if (df is None):
        raise ValueError("df can't be None!")
    if (df.empty):
        raise ValueError("df can't be empty!")

    if(field_to_one_hot_encode not in df.columns):
        raise ValueError('field_to_one_hot_encode must be in df.columns')

    prefix_to_add = prefix if prefix is not None else field_to_one_hot_encode

    df = df.copy()

    # does the one hot encoding on the col
    dummies = pd.get_dummies(df[field_to_one_hot_encode], prefix=prefix_to_add)
    df = pd.concat([df, dummies ], axis=1)

    return df

#
def Populate_df1_in_df2_structure(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame: #TESTED
    """
    Converts df1's column structure (columns and column order) to match df2.
    Missing columns are filled with NaN; extra columns are dropped.
    :param df1: The df we want to be like df2.
    :param df2: The df we usq as the wanted structure.
    :return: Df1 after changing it to be like the df2 structure.
    """
    if (df1 is None):
        raise ValueError("df1 can't be None!")
    if (df1.empty):
        raise ValueError("df1 can't be empty!")

    if (df2 is None):
        raise ValueError("df2 can't be None!")
    if (df2.empty):
        raise ValueError("df2 can't be empty!")

    # reindex aligns df1's columns to match df2's columns exactly
    return df1.reindex(columns=df2.columns)


# todo: add so it will check the date it adds doesn't exists
def add_row_of_new_df_to_og_df(new_df: pd.DataFrame, og_df: pd.DataFrame) -> pd.DataFrame: #TESTED
    """
    This adds the rows of the new_df to the end of the og_df.
    checks if the df are the same in structure.
    :param new_df: The df we want to add at the end of the og_df.
    :param og_df: The df we want to add to.
    :return: The new df, with the rows of the og_df and the rows of the new_df at the end.
    """
    if (new_df is None):
        raise ValueError("df can't be None!")
    if (new_df.empty):
        raise ValueError("df can't be empty!")

    if (og_df is None):
        raise ValueError("df can't be None!")
    if (og_df.empty):
        raise ValueError("df can't be empty!")

    # 1. Check if they have the exact same columns in the exact same order
    if new_df.columns.tolist() != og_df.columns.tolist():
        raise ValueError("The dataframes must have the exact same columns in the same order.")

    # 2. Use axis=0 to stack them vertically (rows)
    # Note: pd.concat automatically returns a new copy, so og_df.copy() isn't strictly necessary here.
    combined_df = pd.concat([og_df, new_df], axis=0, ignore_index=True)

    return combined_df


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




################################################## The main function ##################################################
# todo: finish writing the main function. go over chat comments.

def main() -> None:
    """
    This code does all thr data prep and cleaning.
    After we run it once, if there is no new data, there is no need to run it again.
    It preps and cleans the data and pickles the results.
    There is no need to pickle after each change, so we comment the pickles.
    I checked before and the changes were fine, so there is no need for "check points".
    NOTICE -> there is a field called "next_day_return", THE MODEL DOESN'T TRAIN ON IT!!!
              it is the target col!!!!
    :return:
    """

    # selecting where the code runs -> locally/ Google-colab
    where_the_code_runs = get_where_the_code_runs()

    # loading the five_thousand_days_data_df, the df we work on.
    five_thousand_days_data_df = load_five_thousand_days_data_df(where_the_code_runs = where_the_code_runs)

    # delete
    # todo: run this part just once to change the names of the cols and save it
    five_thousand_days_data_df.rename(columns={"pref_week":"ret_7",
                                               "daily_return_percentage": "ret_1",
                                               "perf_month": "ret_30"})
    # pickle the data
    five_thousand_days_data_df_path = Access_the_file_path(where_the_code_runs = where_the_code_runs,
                                                           path_local = pickle_five_thousand_days_data_file_path_local,
                                                           path_google_colab = pickle_five_thousand_days_data_file_path_google_colab)

    pickling_func(data = five_thousand_days_data_df, file_path = five_thousand_days_data_df_path)
    # delete

    # calling the function and adding the ret_7 data
    # call once to calculate and add the ret_7.
    # when new rows add, need to run this again.

    # def fast_perf_x_trading_days_ago(df: pd.DataFrame, days_ago: int, col_name: str) -> pd.DataFrame:
    five_thousand_days_data_df = fast_perf_x_trading_days_ago(
        df = five_thousand_days_data_df,
        days_ago = 7,
        col_name = 'ret_7'
    )

    print(five_thousand_days_data_df.head(8))

    # calling the function and adding the pref_month data
    # same here
    five_thousand_days_data_df = fast_perf_x_trading_days_ago(
        df = five_thousand_days_data_df,
        days_ago = 30,
        col_name = 'ret_30'
    )


######################## adding more fields #################################333
################################ works -> add the ret_7 col and pickeld the data ###################################


############ adding more feilds i can compute here ################


# The function is right but really slow, each time filters the df,
# will write the same function but much faster
# def ret_1_calculator(symbol: str, date: pd.Timestamp):
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
#     ret_1 = (closing_price_relevant_date - closing_price_one_trading_day_before)/closing_price_one_trading_day_before
#
#     return ret_1


#     adding the now cols and pickling again -> run once
    five_thousand_days_data_df = relative_field_x_days(df = five_thousand_days_data_df,num_of_days = 20,
                                                       new_col_name = "relative_volume_20_days",
                                                       relevant_col_name="volume") # -> checked, it is right

    """
    (df: pd.DataFrame, num_of_days: int,new_col_name: str, relevant_col_name: str)
    """
#
#   adding the ret_2, ret_3, ret_5, ret_10
    five_thousand_days_data_df = add_days_ago_return_percentage_fast(df = five_thousand_days_data_df,
                                                                     days_ago = 2,
                                                                     new_col_name = 'ret_2')
    five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,
                                                                     days_ago =3,
                                                                     new_col_name = 'ret_3')
    five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,
                                                                     days_ago =5,
                                                                     new_col_name = 'ret_5')
    five_thousand_days_data_df = add_days_ago_return_percentage_fast(five_thousand_days_data_df,
                                                                     days_ago =10,
                                                                     new_col_name = 'ret_10')
#
#
#     the name of the current cols:
#     Index(['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low',
#            'close', 'volume', 'ret_7', 'ret_1', 'ret_30',
#            'relative_volume_20_days'],
#           dtype='object')
#
#
#     one hot encoding but keeping the 'symbol' col
    five_thousand_days_data_experiment_df = one_hot_encoding(df = five_thousand_days_data_df,
                                                             field_to_one_hot_encode = 'symbol',
                                                             prefix = 'symbol')
#
#     printing the unique values of the 'exchange' and 'currency'
#     print(five_thousand_days_data_df['exchange'].unique()) -> checked and all the stocks are from ['NASDAQ' 'NYSE']
#     print(five_thousand_days_data_df['currency'].unique()) -> checked and all the stocks price is in USD
#
    five_thousand_days_data_experiment_df = one_hot_encoding(df = five_thousand_days_data_experiment_df,
                                                             field_to_one_hot_encode = 'exchange',
                                                             prefix = 'exchange')
#
#     we saved the last version on the pickle_file_path as well, and saved it on the main df name
    # seleting where to pickle the data based on where we run
    experiment_pickle_file_path = Access_the_file_path(where_the_code_runs = where_the_code_runs,
                                                       path_local = experiment_pickle_file_path_local,
                                                       path_google_colab = experiment_pickle_file_path_google_colab
                                                       )


#   called once and pickled
    five_thousand_days_data_experiment_df = calculate_rolling_mean_for_x_days(df = five_thousand_days_data_experiment_df,
                                                                              field_name = 'close',num_of_days = 20,
                                                                              new_col_name = 'SMA_20',
                                                                              after_col_put_new_col = 'ret_10')

    five_thousand_days_data_experiment_df = calculate_rolling_mean_for_x_days(df = five_thousand_days_data_experiment_df,
                                                                              field_name = 'close',
                                                                              num_of_days = 50,
                                                                              new_col_name = 'SMA_50',
                                                                              after_col_put_new_col = 'SMA_20')
    print(five_thousand_days_data_experiment_df.iloc[40:60])
    print(five_thousand_days_data_experiment_df.columns.get_loc('ret_10'))


    five_thousand_days_data_experiment_df = add_sma_x_gap_percent(five_thousand_days_data_experiment_df, 'SMA_20', 'SMA_20_gap_percent', 'SMA_50')
    five_thousand_days_data_experiment_df = add_sma_x_gap_percent(five_thousand_days_data_experiment_df, 'SMA_50', 'SMA_50_gap_percent', 'SMA_20_gap_percent')
    # pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved



    five_thousand_days_data_experiment_df = five_thousand_days_data_experiment_df.copy()
    five_thousand_days_data_experiment_df["date"] = pd.to_datetime(five_thousand_days_data_experiment_df["date"])
    five_thousand_days_data_experiment_df = (
        five_thousand_days_data_experiment_df
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    # # same-day return: keep as feature
    five_thousand_days_data_experiment_df["ret_1"] = (
        five_thousand_days_data_experiment_df.groupby("symbol")["close"].pct_change(1)
    )

    """
    NOTICE -> next_day_return IS THE TARGET COL, THE MODELS DO NOT TRAIN ON IT.
    """
    # # true next-day target -> this is a ratio not percentage
    # This is -> (tomorrow's Close/ Today's Close) - 1
    # the model doesn't train on this !!!!!! this is the y col.
    five_thousand_days_data_experiment_df["next_day_return"] = (
        five_thousand_days_data_experiment_df.groupby("symbol")["close"].shift(-1)
        / five_thousand_days_data_experiment_df["close"] - 1
    )

    ################################### spliting the data to train_val_set and test_set###################################
    #
    #
    # print(five_thousand_days_data_experiment_df.head(3))
    #
    # # from the test i checked, it fucked up the XGBoost model, so drop it for now
    # also we protect it from the error of the col not existing
    # five_thousand_days_data_experiment_df = five_thousand_days_data_experiment_df.drop(
    #     columns=['intraday_range_percent'],
    #     errors='ignore'
    # )


    # here we made sure all the stocks have the same starting date
    five_thousand_days_data_experiment_df = same_start_date_for_all_stocks(five_thousand_days_data_experiment_df)

    # we make sure all symbols have the same dates
    five_thousand_days_data_experiment_df = keep_common_dates_only(five_thousand_days_data_experiment_df)
    pickling_func(five_thousand_days_data_experiment_df, experiment_pickle_file_path) # called it once, now it is saved


    five_thousand_days_data_experiment_df = load_five_thousand_days_data_experiment_df(
        where_the_code_runs = where_the_code_runs
    )

    # we will take 85% of the data to be in the train_validation set,
    # and the other 15% we will take for the test set -> so we won't train our model on the test set.
    train_set, validation_set, data_experiment_test_df = split_df_to_train_val_test(df = five_thousand_days_data_experiment_df,
                                                                     percent_for_train = 70,
                                                                     percent_for_val = 15,
                                                                     percent_for_test = 15
                                                                    )


    data_experiment_train_and_validation_df = pd.concat([train_set, validation_set], ignore_index=True)
    #
    # # it works
    # print(f'data_experiment_train_and_validation_df: {len(data_experiment_train_and_validation_df)}')
    # print(f'data_experiment_test_df: {len(data_experiment_test_df)}')
    # print(f'five_thousand_days_data_experiment_df: {len(five_thousand_days_data_experiment_df)}\n')

    # selecting the path based on where we run the code
    experiment_train_and_validation_pickle_file_path = Access_the_file_path(where_the_code_runs = where_the_code_runs,
                                                                            path_local = experiment_train_and_validation_pickle_file_path_local,
                                                                            path_google_colab = experiment_train_and_validation_pickle_file_path_google_colab)

    data_experiment_train_and_validation_df = keep_common_dates_only(data_experiment_train_and_validation_df)
    pickling_func(data_experiment_train_and_validation_df, experiment_train_and_validation_pickle_file_path) # called it once, now it is saved
    data_experiment_train_and_validation_df = load_data_experiment_train_and_validation_df(where_the_code_runs = where_the_code_runs)

    print(
        f'list of the columns of data_experiment_train_and_validation_df: {data_experiment_train_and_validation_df.columns.tolist()}'
    )

    # selecting the path
    experiment_test_pickle_file_path = Access_the_file_path(
        where_the_code_runs=where_the_code_runs,
        path_local=experiment_test_pickle_file_path_local,
        path_google_colab=experiment_test_pickle_file_path_google_colab
    )

    data_experiment_test_df = keep_common_dates_only(data_experiment_test_df)
    pickling_func(data_experiment_test_df, experiment_test_pickle_file_path) # called it once, now it is saved


    ###################################################checked -> works###################################################
    # print(data_experiment_test_df.columns.tolist())
    # # this prints the amount of rows each stock has
    # print(data_experiment_test_df.groupby('symbol').size().sort_values())
    # print(data_experiment_test_df['symbol'].nunique())
    # print(data_experiment_train_and_validation_df['symbol'].nunique())
    # print(data_experiment_train_and_validation_df.groupby('symbol').size().sort_values())
    ###################################################checked -> works###################################################

    # print the min max (biggest min - the min date that all symbols have)
    min_date_all_symbols_have = data_experiment_train_and_validation_df.groupby('symbol')[
        'date'].min().max()  # 2020-09-30 00:00:00
    max_date = data_experiment_train_and_validation_df.groupby('symbol')['date'].max().max()  # 2025-05-20 00:00:00
    print(f"min date that all symbols have: {min_date_all_symbols_have}")
    print(f"max date in the train_val_set: {max_date}\n")

    # delete
    print('data_experiment_test_df: ')
    print(data_experiment_test_df.columns.tolist())

    print('\ndata_experiment_train_and_validation_df: ')
    print(data_experiment_train_and_validation_df.columns.tolist())




    # delete


# running the code
if __name__ == "__main__":
    main()



