"""
To handles all the API calls logic to the "twelve data" website.
"""

# import WakaTime
# email
# regulr password
from __future__ import annotations

import time
import requests
import pandas as pd
from twelvedata import TDClient

################ Twelve data this is a website for API calls for stocks data ############################

"""
we need to call the API calls just once,
after that we just pickle the data in a file and then
just call it -> FOR THE START DATA ONLY!!!
"""

# todo: write the API key in a different place
Twelve_data_API_key = '8b8436f89bc649d1921065d6bfca8c60'

# Initialize client with your API key
td = TDClient(apikey=Twelve_data_API_key)

# todo: this is the API doc: https://twelvedata.com/docs#ws-real-time-price, use it


"""
A list of all the acceptable_intervals
"""
acceptable_intervals = ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week', '1month']


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
    Loops over a list of symbols and returns a dict with -> key: symbol, value: Earliest Timestamp
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


# todo: Maybe need to fix here as well the part of the start date
def company_data_prices_fetcher(ticker_symbol: str, wanted_interval: str, how_many_intervals: int,
                                start_date:  pd.Timestamp |None = None, end_date: pd.Timestamp |None = None)\
                                -> pd.DataFrame:
    """
    This function returns the Price bars data
    NOTICE: the error handling is built-in the API.
    :param ticker_symbol: the symbol of the stock
    :param wanted_interval: the wanted interval: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    :param how_many_intervals: the number of calls.
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
                   companies_list_earliest_timestamp_dict: dict[str, pd.Timestamp] | None = None,
                   start_date:  pd.Timestamp |None = None, end_date: pd.Timestamp |None = None,
                   forward_or_backward_flag: int | None = None,
                   check_Earliest_Timestamp_for_df_flag: int | None = None) -> pd.DataFrame:
    """
    Returns a pandas df with the data of all the companies in company_names_list.
    NOTICE: the error handling is built-in the API.
    :param ticker_list: List of the names of the relevant companies
    :param wanted_interval: Interval -> 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    :param how_many_intervals: How many interval records do we want
            NOTICE: the range of is [1,5000]
            NOTICE: if you want daily prices, hour prices, and min for x days you need to so.
            NOTICE: if you use the function in different days, the time frame will shift as well
    :param companies_list_earliest_timestamp_dict: A dict with earliest timestamp per symbol.
    :param start_date: The code won't take any rows before this date. None -> most recent date in the df,
                       not necessarily today.
    :param end_date: The code won't take any rows after this date. None -> most recent date in the df,
                       not necessarily today.
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

    if(check_Earliest_Timestamp_for_df_flag == 1):
        # a dict of the earliest_timestamp of all the companies in the given df
        companies_list_earliest_timestamp_dict = companies_list_earliest_timestamp_dict_fetcher(
            ticker_list,
            wanted_interval
        )

    # we need the most recent date or the date we want to get
    if companies_list_earliest_timestamp_dict is None and start_date is not None:
        raise ValueError(
            'companies_list_earliest_timestamp_dict must be provided when start_date is used.'
        )


    # the api allows 8 ticker in a min or less, so we added a counter
    # after 8 companies, we sleep for a min
    for i, ticker in enumerate(ticker_list, start=1):
        # if we are getting data from today backwards, forward_or_backward_flag is None ->
        # there is no need to check what is the Earliest Timestamp.
        # if we reach it we will stop.
        # if we want to start for date x and go forward, forward_or_backward_flag -> we need to now what is the Earliest Timestamp.

        ticker_earliest_timestamp = None

        if companies_list_earliest_timestamp_dict is not None:
            if ticker not in companies_list_earliest_timestamp_dict:
                raise ValueError('This symbol does not exists in the companies_list_earliest_timestamp_dict!')

            ticker_earliest_timestamp = companies_list_earliest_timestamp_dict[ticker]


        # so we won't override the start_date, end_date for all the symbols
        # each symbol we check separately
        ticker_start_date = start_date
        ticker_end_date = end_date
        if (
                ticker_earliest_timestamp is not None
                and ticker_start_date is not None
                and ticker_start_date < ticker_earliest_timestamp
        ):
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

