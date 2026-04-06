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
from data_preparing import (pickling_func,unpickle_data, company_exchange_and_currency_fetcher,
                            company_data_prices_fetcher, all_df_creator, adding_col_with_values,
                            move_col_position_in_df, top_40_tech_names, five_thousand_days_data_df,
                            company_earliest_timestamp_fetcher, same_start_date_for_all_stocks,
                            Earliest_Timestamps_top_40_tech_companies_daily)

from evaluation_and_simulation import where_the_code_runs

"""
We will create a df with hourly prices for the 40 stocks here.
Later we will add these columns to the daily prices df to create one
big df.
NOTICE: We will use some of the functions for data_preparing.py.
NOTICE: We might need to make changes to the models and function in evaluation_and_simulation.py after the merge.
"""



"""
Getting the hourly prices data for the last 5000 hours -> how_many_intervals = 5,000 * 24 = 120000
"""
# 5000 is the max number of Requests
# in this df there is the all the rows
min_start_date = five_thousand_days_data_df['date'].min() #2006-05-01 00:00:00
min_start_date_plus_five_thousand_hours = pd.to_datetime('2006-11-25')
print(min_start_date)
current_start_date = pd.to_datetime(min_start_date)
five_thousand_hourly_data = all_df_creator(top_40_tech_names, '1h',
                                           5000, start_date = current_start_date,
                                           end_date = min_start_date_plus_five_thousand_hours) # called it once, now it is saved

# this is for temp files
pickle_hourly_temp_file_final= r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_final_data.pkl"
# pickling the data for future use
pickle_hourly_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_data.pkl"
# this is for temp files
pickle_hourly_temp_file_path = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_temp_data.pkl"

# run on google colab
if(where_the_code_runs == 2):
    pickle_hourly_temp_file_path = r"/content/algo_trading_stocks/data/five_thousand_hourly_temp_data.pkl"
    pickle_hourly_file_path = r"five_thousand_hourly_data.pkl"

"""
the daily df has col->'date'->2020-09-30
we here have a col->'datetime'-> 2026-04-02 15:30:00
so we have to: 
1. create a 'date' col in five_thousand_hourly_data_df.
2. put only the date in it.
3. pickle again, easy access.
"""

# 1 + 2 + 3:
# five_thousand_hourly_data['datetime'] = pd.to_datetime(five_thousand_hourly_data['datetime'])
# five_thousand_hourly_data['date'] = five_thousand_hourly_data['datetime'].dt.date
# pickling_func(five_thousand_hourly_data, pickle_hourly_file_path) # called it once, now it is saved
first_five_thousand_hourly_data_df = unpickle_data(pickle_hourly_file_path)
print(first_five_thousand_hourly_data_df)


"""
Add alot more rows to the df
"""


def create_df_of_hourly_prices(df: pd.DataFrame):
    """
    Uses the function from the data_preparing to get the hourly prices.
    :param df: The df we want to add hourly prices to.
    :return: A df with hourly prices of the stocks in the df for the same dates as in the df.
    """

    final_df = df.copy()
    # adding the date col
    final_df['datetime'] = pd.to_datetime(final_df['datetime'])
    final_df['date'] = final_df['datetime'].dt.date

    df_fields = df.columns.tolist()
    if not (('datetime' in df_fields) and ('symbol' in df_fields)):
        raise ValueError('Both date and symbol must be in the df!')

    # the last date we want to get the hourly prices for
    # it is the max date of the df
    end_date_final = pd.to_datetime(final_df['date']).max()
    # same idea for the start date, min
    start_date_final = (final_df['datetime']).min()

    # the start date each iteration changes
    current_start_date = pd.to_datetime(start_date_final) + pd.Timedelta(hours=5000)
    current_end_date = current_start_date + pd.Timedelta(hours=4999)
    # first call
    hourly_prices_df = all_df_creator(
        final_df['symbol'].unique().tolist(),
        '1h',
        5000,
        start_date=current_start_date,
        end_date = current_end_date
    )

    # adding the date col
    hourly_prices_df['datetime'] = pd.to_datetime(hourly_prices_df['datetime'])
    hourly_prices_df['date'] = hourly_prices_df['datetime'].dt.date

    final_df = pd.concat([final_df, hourly_prices_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['symbol', 'datetime']).sort_values(['symbol', 'datetime']).reset_index(
        drop=True)

    current_start_date = pd.to_datetime(hourly_prices_df['datetime']).max() + pd.Timedelta(hours=1)
    current_end_date = current_start_date + pd.Timedelta(hours=5000)
    # just to see the progress
    counter = 1
    # make calls until we get to the final date_and_hour
    while pd.to_datetime(hourly_prices_df['datetime']).max() < pd.to_datetime(end_date_final).normalize():
        # just to see the progress
        max_hourly_prices_date = pd.to_datetime(hourly_prices_df['datetime']).max()
        if (counter % 10 == 0):
            print(f'counter: {counter}')
            print(f'hourly_prices_df max: {max_hourly_prices_date}')
            print(f'end_date_final: {end_date_final}')

        # i do not think we need to add a 'sleep(60)' here because there is one in the all_df_creator function
        ####
        # so we won't hard code the stocks: top_40_tech_names -> top_40_tech_names

        hourly_prices_df = all_df_creator(final_df['symbol'].unique().tolist(), '1h',5000,
                                          start_date = current_start_date,
                                          end_date = current_end_date)
        # adding the date col
        hourly_prices_df['datetime'] = pd.to_datetime(hourly_prices_df['datetime'])
        hourly_prices_df['date'] = hourly_prices_df['datetime'].dt.date

        final_df = pd.concat([final_df, hourly_prices_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['symbol', 'datetime']).sort_values(
            ['symbol', 'datetime']).reset_index(drop=True)
        # current start -> max_date_and_hour + 1 hour
        current_start_date = pd.to_datetime(hourly_prices_df['datetime']).max() + pd.Timedelta(hours=1)
        # current end -> current_start_date + 5000 hour
        current_end_date = current_start_date + pd.Timedelta(hours=5000)
        counter += 1

    return final_df



final_df = create_df_of_hourly_prices(first_five_thousand_hourly_data_df)

pickling_func(final_df, pickle_hourly_file_path) # called it once, now it is saved
final_five_thousand_days_of_hourly_data_df = unpickle_data(pickle_hourly_file_path)

print(first_five_thousand_hourly_data_df)
print('final_five_thousand_days_of_hourly_data_df:\n')
print(final_five_thousand_days_of_hourly_data_df)
















