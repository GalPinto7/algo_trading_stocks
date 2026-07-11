"""
This code is responsible for:
1. getting these fields:
    ['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low', 'close', 'volume']
    from the API, and the rest of the cols it calculates.
NOTE: FIELDS CAN ADD OR DELETED, SO CODE MIGHT NEED MAINTENANCE.

2. Cleaning the new data.

3. Append the new data to the csv and update old data where needed.

4. Calculate the values of the cols we can't get from the API to the new rows.

5. Saves the new df.
"""

# Todo: understand what is the best time to run this code.

from __future__ import annotations

from datetime import datetime, timedelta, date
# import WakaTime
from data_preparing import (
    load_top_40_tech_companies_names,
    load_Earliest_Timestamps_top_40_tech_companies_daily,
    one_hot_encoding,
    fast_perf_x_trading_days_ago,
    Populate_df1_in_df2_structure,
    load_five_thousand_days_data_df,
    add_row_of_new_df_to_og_df,
    pickling_func,
    Access_the_file_path,
    pickle_five_thousand_days_data_file_path_local,
    pickle_five_thousand_days_data_file_path_google_colab,
    pickle_five_thousand_days_data_file_path_for_testing_local,
    pickle_five_thousand_days_data_file_path_for_testing_google_colab,
    fast_perf_x_trading_days_ago_for_last_date,
    relative_field_x_days_for_last_date,
    calculate_rolling_mean_for_x_days_for_last_date
)
from twelve_data_api import all_df_creator
from runtime_config import get_where_the_code_runs
import pandas as pd
import numpy as np

# 2026-05-23

# todo: make sure we have all the needed cols.
# todo: make sure the cols are in the correct order.
# todo: understand when to run this code, what time of the day.
# todo: make sure the code runs auto each day.
def get_most_recent_data(companies_names: list[str], companies_list_earliest_timestamp_dict: dict) -> pd.DataFrame:
    """
    Calling to receive the data of the companies for most recent data in the website. Not necessarily today.
    fields returned: symbol, exchange, currency, date, open, high, low, close, volume, one hot encoding of exchange,
    one hot encoding of symbol.
    NOTICE -> we can't creat all the cols here, because we need some historic data for it.
    Only after we merge this data with the old one, we can calculate the rest.
    The fields this function returns: [
        "symbol",
        "exchange",
        "currency",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        one hot encoding of exchange,
        one hot encoding of symbol
    ]
    :param companies_names: A list of the companies we want the data on.
    :param companies_list_earliest_timestamp_dict: A dict.
    :return: A pandas df with the data of the most recent records of these stocks in the website.
    """
    if(companies_list_earliest_timestamp_dict is None):
        raise ValueError("companies_list_earliest_timestamp_dict can't be None.")
    elif(companies_names is None):
        raise ValueError("companies_names can't be None.")
    elif len(companies_names) == 0:
        raise ValueError("companies_names can't be empty.")
    elif companies_list_earliest_timestamp_dict == {}:
        raise ValueError("companies_list_earliest_timestamp_dict can't be empty.")

    today_date = pd.Timestamp.today().normalize()

    most_recent_data = all_df_creator(
        ticker_list= companies_names,
        wanted_interval= "1day",
        how_many_intervals= 1,
        companies_list_earliest_timestamp_dict= companies_list_earliest_timestamp_dict
    )

    if most_recent_data.empty:
        return most_recent_data

    if "date" not in most_recent_data.columns:
        if "datetime" in most_recent_data.columns:
            most_recent_data = most_recent_data.rename(columns={"datetime": "date"})
        else:
            raise ValueError("most_recent_data must contain either 'date' or 'datetime' column.")

    most_recent_data["date"] = pd.to_datetime(most_recent_data["date"]).dt.normalize()

    # We require all received rows to be from today.
    # If even one row is not today, return empty df.
    if not (most_recent_data["date"] == today_date).all():
        return most_recent_data.iloc[0:0].copy()

    # adding more cols to the df

    # one hot encoding for the exchange
    most_recent_data = one_hot_encoding(df=most_recent_data,
                                        field_to_one_hot_encode='exchange',
                                        prefix='exchange')
    # one hot encoding for the symbol
    most_recent_data = one_hot_encoding(df = most_recent_data,
                                        field_to_one_hot_encode = "symbol",
                                        prefix = "symbol")

    return most_recent_data


# todo: should I move it to data_preparing.py?

def extract_new_rows_from_full_df(full_df: pd.DataFrame, new_row_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the new rows from a full df, based on ['symbol', 'date'].

    :param full_df: The full dataframe: old rows + new rows.
    :param new_row_keys: DataFrame with ['symbol', 'date'] of only the new rows.
    :return: Only the new rows from full_df.
    """

    if full_df is None:
        raise ValueError("full_df can't be None.")
    if new_row_keys is None:
        raise ValueError("new_row_keys can't be None.")

    if full_df.empty:
        raise ValueError("full_df can't be empty.")
    if new_row_keys.empty:
        raise ValueError("new_row_keys can't be empty.")

    required_cols = ["symbol", "date"]

    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"{col} must be in full_df.")
        if col not in new_row_keys.columns:
            raise ValueError(f"{col} must be in new_row_keys.")

    full_df = full_df.copy()
    new_row_keys = new_row_keys.copy()

    full_df["date"] = pd.to_datetime(full_df["date"]).dt.normalize()
    new_row_keys["date"] = pd.to_datetime(new_row_keys["date"]).dt.normalize()

    return full_df.merge(
        new_row_keys,
        on=["symbol", "date"],
        how="inner"
    )

def calculate_values_of_rest_of_cols(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates only the ret_n columns for the new rows.

    Current columns calculated:
    ret_1, ret_2, ret_3, ret_5, ret_7, ret_10, ret_30

    It keeps all old rows unchanged.
    Later we will create separate functions for SMA, relative volume, RSI, etc.
    """

    if old_df is None:
        raise ValueError("old_df can't be None.")
    if new_df is None:
        raise ValueError("new_df can't be None.")

    if old_df.empty:
        raise ValueError("old_df can't be empty.")
    if new_df.empty:
        raise ValueError("new_df can't be empty.")

    required_cols = ["symbol", "date", "close"]

    for col in required_cols:
        if col not in old_df.columns:
            raise ValueError(f"{col} must be in old_df.")
        if col not in new_df.columns:
            raise ValueError(f"{col} must be in new_df.")

    old_df = old_df.copy()
    new_df = new_df.copy()

    old_df["date"] = pd.to_datetime(old_df["date"]).dt.normalize()
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.normalize()

    # Make sure the new rows have the same structure as the old df.
    current_new_rows_df = Populate_df1_in_df2_structure(
        df1=new_df,
        df2=old_df
    )

    # Save the identity of the new rows.
    # We use this after every calculation to extract only the new rows again.
    new_row_keys = current_new_rows_df[["symbol", "date"]].copy()

    result_df = old_df.copy()

    ret_cols_and_days = [
        ("ret_1", 1),
        ("ret_2", 2),
        ("ret_3", 3),
        ("ret_5", 5),
        ("ret_7", 7),
        ("ret_10", 10),
        ("ret_30", 30),
    ]

    for col_name, days_ago in ret_cols_and_days:
        result_df = fast_perf_x_trading_days_ago_for_last_date(
            new_day_df=current_new_rows_df,
            historic_df=old_df,
            days_ago=days_ago,
            name_of_col=col_name
        )

        # Important:
        # After calculating one ret_n column, take the updated new rows
        # so the next ret_n calculation does not erase the previous one.
        current_new_rows_df = extract_new_rows_from_full_df(
            full_df=result_df,
            new_row_keys=new_row_keys
        )

    return result_df







# date format : 2006-05-01

def main() -> None:
    """
    Run the daily prices update manually.
    1. Gets the daily data of our stocks. -> DONE
    2. Changes the structure of the new dd to be like the og df. -> DONE
    3. Adds the rows of the new df to the og df. -> DONE
    3. For the new records, calculate the values of the cols we can't get from the API.
    4. Saves the new df instead of the og df. PKL.

    notes:
    * We work under the assumption this code runs every day.

    """
    today_date = pd.Timestamp.today().normalize()

    """
    This code run automatically everyday at time ___.
    This code runs on my local pc only.
    
    """
    where_the_code_runs = get_where_the_code_runs()

    top_40_tech_names = load_top_40_tech_companies_names(where_the_code_runs)

    Earliest_Timestamps_top_40_tech_companies_daily = load_Earliest_Timestamps_top_40_tech_companies_daily(
        where_the_code_runs)

    new_daily_data = get_most_recent_data(
        companies_names=top_40_tech_names,
        companies_list_earliest_timestamp_dict=Earliest_Timestamps_top_40_tech_companies_daily
    )

    # break condition if there aren't any new rows.
    if new_daily_data.empty:
        print("No new daily data available. Nothing was updated.")
        return

    # upload the og df, so we could compare the structure.
    five_thousand_days_data_df = load_five_thousand_days_data_df(where_the_code_runs=where_the_code_runs)

    # Changes the structure of the new dd to be like the og df
    new_daily_data = Populate_df1_in_df2_structure(df1=new_daily_data,df2=five_thousand_days_data_df)

    # adds the row of new_daily_data to the end of five_thousand_days_data_df
    five_thousand_days_data_df = add_row_of_new_df_to_og_df(new_df=new_daily_data,og_df=five_thousand_days_data_df)

    # todo: calculate the values of the missing cols
    # we need to calcukate


    # selecting the relevant file path based on where do we run the code
    # TODO: NOTICE for now we save the new df in a new pkl file so we won't delete the old one,
    #  we do this util we are sure this code works.
    five_thousand_days_data_df_path = Access_the_file_path(where_the_code_runs=where_the_code_runs,
                                                           path_local=pickle_five_thousand_days_data_file_path_for_testing_local,
                                                           path_google_colab=pickle_five_thousand_days_data_file_path_for_testing_google_colab
                                                       )

    # Saves the new df
    pickling_func(data = five_thousand_days_data_df, file_path=five_thousand_days_data_df_path)



if __name__ == "__main__":
    main()





"""
fields we return:
['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low', 'close', 'volume']
all needed fields:
['symbol', 'exchange', 'currency', 'date', 'open', 'high', 'low', 'close', 'volume', 
'pref_week', 'daily_return_percentage',
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







