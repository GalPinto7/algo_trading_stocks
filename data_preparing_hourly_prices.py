import pandas as pd

from data_preparing import (
    pickling_func,
    unpickle_data,
    load_top_40_tech_companies_names,
    load_five_thousand_days_data_df,
    load_Earliest_Timestamps_top_40_tech_companies_hourly,
    Access_the_file_path
)
from twelve_data_api import all_df_creator
from runtime_config import get_where_the_code_runs

"""
We will create a df with hourly prices for the 40 stocks here.
Later we will add these columns to the daily prices df to create one
big df.
NOTICE: We will use some of the functions for data_preparing.py.
NOTICE: We might need to make changes to the models and function in evaluation_and_simulation.py after the merge.
"""




"""
Add alot more rows to the df
"""



#####################################################paths of files#####################################################
# this is for final files
pickle_hourly_temp_file_final_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_final_data.pkl"
pickle_hourly_temp_file_final_google_colab = r"/content/algo_trading_stocks/data/five_thousand_hourly_final_data.pkl"


# pickling the data for future use
pickle_hourly_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_data.pkl"
pickle_hourly_file_path_google_colab = r"/content/algo_trading_stocks/data/five_thousand_hourly_data.pkl"

# this is for temp files
pickle_hourly_temp_file_path_local = r"C:\Users\galpi\Desktop\stocks algo trading - 14.03.2026\data\five_thousand_hourly_temp_data.pkl"
pickle_hourly_temp_file_path_google_colab = r"/content/algo_trading_stocks/data/five_thousand_hourly_temp_data.pkl"
########################################################################################################################


def create_df_of_hourly_prices(df: pd.DataFrame, companies_list_earliest_timestamp_dict: dict):
    """
    Uses the function from the data_preparing to get the hourly prices.
    :param df: The df we want to add hourly prices to.
    :param companies_list_earliest_timestamp_dict: A dict with the earliest_timestamp od each of the 40 companies.
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
        wanted_interval='1h',
        how_many_intervals= 5000,
        companies_list_earliest_timestamp_dict = companies_list_earliest_timestamp_dict,
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
        hourly_prices_df = all_df_creator(ticker_list = final_df['symbol'].unique().tolist(),wanted_interval = '1h',
                                          how_many_intervals = 5000,
                                          companies_list_earliest_timestamp_dict = companies_list_earliest_timestamp_dict,
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



def main() -> None:
    """
    Getting the hourly prices data for the last 5000 hours
    """
    # running locally or on Google colab
    where_the_code_runs = get_where_the_code_runs()
    # 5000 is the max number of Requests
    # in this df there is the all the rows
    five_thousand_days_data_df = load_five_thousand_days_data_df(where_the_code_runs = where_the_code_runs)

    # getting the companies_list_earliest_timestamp_dict
    Earliest_Timestamps_top_40_tech_companies_hourly  = load_Earliest_Timestamps_top_40_tech_companies_hourly(
        where_the_code_runs = where_the_code_runs,
    )

    # getting the min date
    min_start_date = five_thousand_days_data_df['date'].min()  # 2006-05-01 00:00:00
    min_start_date_plus_five_thousand_hours = pd.to_datetime('2006-11-25')
    print(min_start_date)

    # getting the names of the 40 companies from the pickled file
    top_40_tech_names = load_top_40_tech_companies_names(where_the_code_runs)
    current_start_date = pd.to_datetime(min_start_date)
    five_thousand_hourly_data = all_df_creator(top_40_tech_names, '1h',
                                               how_many_intervals = 5000,
                                               companies_list_earliest_timestamp_dict = Earliest_Timestamps_top_40_tech_companies_hourly,
                                               start_date=current_start_date,
                                               end_date=min_start_date_plus_five_thousand_hours)  # called it once, now it is saved

    # selecting the right path -> local/colab
    path_hourly_data = Access_the_file_path(where_the_code_runs = where_the_code_runs,
                                path_local = pickle_hourly_file_path_local,
                                path_google_colab = pickle_hourly_file_path_google_colab)

    """
    the daily df has col->'date'->2020-09-30
    we here have a col->'datetime'-> 2026-04-02 15:30:00
    so we have to: 
    1. create a 'date' col in five_thousand_hourly_data_df.
    2. put only the date in it.
    3. pickle again, easy access.
    """

    # 1 + 2 + 3:
    five_thousand_hourly_data['datetime'] = pd.to_datetime(five_thousand_hourly_data['datetime'])
    five_thousand_hourly_data['date'] = five_thousand_hourly_data['datetime'].dt.date
    pickling_func(five_thousand_hourly_data, path_hourly_data) # called it once, now it is saved
    first_five_thousand_hourly_data_df = unpickle_data(path_hourly_data)
    print(first_five_thousand_hourly_data_df)

    final_df = create_df_of_hourly_prices(df = first_five_thousand_hourly_data_df,
                                          companies_list_earliest_timestamp_dict = Earliest_Timestamps_top_40_tech_companies_hourly
                                          )

    # # selecting the right path for the final df-> local/colab
    # todo: fix later when this file is in the GitHub
    path_hourly_data_final = Access_the_file_path(where_the_code_runs=where_the_code_runs,
                                            path_local=pickle_hourly_temp_file_final_local,
                                            path_google_colab=pickle_hourly_temp_file_final_google_colab)

    pickling_func(final_df, path_hourly_data_final)  # called it once, now it is saved
    final_five_thousand_days_of_hourly_data_df = unpickle_data(path_hourly_data_final)

    print(first_five_thousand_hourly_data_df)
    print('final_five_thousand_days_of_hourly_data_df:\n')
    print(final_five_thousand_days_of_hourly_data_df)





"""
If we want to change the  
"""
if __name__ == '__main__':
    main()






