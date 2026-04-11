from __future__ import annotations
from typing import Optional

from typing import Any, Tuple, Dict
from sklearn.metrics import zero_one_loss
import numpy as np
from numpy import floating
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from datetime import datetime, timedelta, date
from typing import Any, List, Dict, Tuple, Union
from typing import TypedDict, Dict
from models.base_line_models import DumbModel, PreviousDayReturnModel,RollingAvgModel
from models.model_base import Model
from models.ml_models import XGBoostModel, LinearRegressionModel
from models.neural_network_models import FullyConnectedNeuralNetwork
import pandas as pd
from data_preparing import (pickle_file_path,unpickle_data, experiment_pickle_file_path, min_date_all_symbols_have,
                            max_date, relative_field_x_days, experiment_train_and_validation_pickle_file_path, 
                            experiment_test_pickle_file_path, data_split_to_train_and_validation)
from runtime_config import where_the_code_runs


"""
We write the code with 'if __name__ == "__main__":', so all the class + def we'll keep
outside the 'if __name__ == "__main__":' block, but the things we want to run each time we do the simulation,
we will write inside the block.
All the executable code -> inside the block.
All the rest -> not inside the block.
"""

# multy threads?
# todo: if you are sure the changes you do are correct -> push to git -> update files in google colab.
# todo: if you add a library -> add it to the requirements.txt file.
# todo: scale the data -> do it per iteration, if you do it on all the data
#  that will be leakage, because than the price t can now the max price ever, that
#  may still not happened, so he knows it will happen in the future -> data leakage.

# todo: another thing the chat said to fix.

# todo: something looks off -> 1. linear regression bought and sold but no there was no different in the money.
# todo: the XGBoost used to make money, now losses?
"""
We may add more fields to the table.
We use regression because we predict the price -> continuous.
We here compare the different models.
We moved to object oriented programming.
"""
# todo: add checks to the code -> a lot.
# todo: read the last conv with chat and add the features.
# todo: for now close is fine, in the future must change, maybe to 'open' -> can't but when close.
# todo: maybe add exception handling in functions
# XGBoost knows how to handle booleans


# todo: you can move this to an enum class
# todo: update this each time you add a new model
# a dict of the relevant_models_numbers_for_x_scaling -> update each time we add a new relevant model.
# flag == 0 -> dumb model: no scaling ,
# flag == 1 -> XGBoost model: no scaling
# flag == 2 -> previous daily return pred: no scaling,
# flag == 3 -> rolling AVG pred: no scaling
# flag == 4 -> Linear regression: scaling
# flag == 5 -> FullyConnectedNeuralNetwork: scaling
relevant_models_and_numbers_dict = {0: 'dumb_model', 1: 'XGBoost_model', 2:'previous_daily_return',
                                         3: 'rolling_AVG_pred', 4:'Linear_regression', 5: 'FullyConnectedNeuralNetwork'}

relevant_models_and_numbers_dict_for_scaling = {4:'Linear_regression', 5: 'FullyConnectedNeuralNetwork'}
# validation_type: Type of test we want to do
# validation_type is None -> AVG RMSE
# validation_type = 2 -> right direction: y_prd * t_validation > 0 <-> 1 O.W 0
# NOT EQUAL TO 0 BECAUSE FOR THE DUMB MODEL y_prd = 0 SO y_prd * t_validation = 0 ALWAYS
# validation_type = 3 -> make guess only if |y_pred| > initial_threshold
# a dict of all the relevant test types and their numbers.
validation_types_dict = {1: 'RMSE', 2: "right_direction",
                         3: "right_direction_over_threshold"}



# create a model to use on all the functions -> OOP
"""
we will create a global array of the flags of the models
"""

array_of_flags_of_models = [0,1,2,3,4,5]
def get_model(flag: int, num_of_days: int | None = None,
              relevant_col_name: str = "daily_return_percentage",
              num_of_x_fields: int | None = None) -> Model:
    """
    Factory method for selecting a model based on flag.
    :param flag: A flag to select which model to use.
    0 -> Dumb model, 1 -> XGB, 2 -> previous_daily_return, 3 -> rolling_avg_baseline, 4 -> Linear regression,
    5 -> FullyConnectedNeuralNetwork,
    :param num_of_days: The number of days we use for the rolling_avg_baseline calculation -> relevant only for this.
    :param relevant_col_name: The column we calculate the rolling_avg_baseline on ->
                              relevant for rolling_avg_baseline and PreviousDayReturnModel
    :param num_of_x_fields: The number of x fields, the fields we train the model on without the target column
    :return: The model we will use.
    """
    if not ((flag in array_of_flags_of_models) or (flag is None)):
        raise ValueError("flag must be 0, 1, 2, 3, 4 or None")

    model = None
    # todo: fix that flags so first flags are base line, second ML, third NN
    if (flag == 0 or flag is None):
        model = DumbModel()
    elif( flag == 1):
        model = XGBoostModel()
    elif( flag == 2):
        model = PreviousDayReturnModel(relevant_col_name = relevant_col_name)
    elif flag == 3:
        if ((num_of_days is None) or (num_of_days <= 0)):
            raise ValueError("num_of_days must be provided and positive when flag == 3")
        return RollingAvgModel(num_of_days=num_of_days, relevant_col_name=relevant_col_name)
    elif flag == 4:
        return LinearRegressionModel()

    # the models from here onwards are NN
    elif flag == 5:
        if ((num_of_x_fields is None) or (num_of_x_fields <= 0)):
            raise ValueError('num of fields must be positive')
        model = FullyConnectedNeuralNetwork(num_of_x_fields) # add number of fields
    else:
        raise ValueError(f"Flag {flag} is not a valid model type.")

    return model





################################### compare to other models on RMSE ###################################
"""
To understand if the model is any good we need to compare it to some thing else.
will make a few tests to see how it compares to them:
0. dumb model - predicts 0 return always -> passed:
(XGBoost model (1) is better than Dumb model->
   AVG RMSE: 0.017893688794979667 < 0.026539167282387834 -> 33% improvement)
   
2. previous daily return -> the model takes the daily return of the previous day of
   the symbol and predicts the next day daily return will be the same -> passed
   XGBoost model is better than previous daily return model:
   AVG RMSE: 0.017893688794979667 < 0.03797977338400326 -> 40% improvement
   (makes sense previous daily return is worst then Dumb model, stocks can stable)

3. rolling AVG test -> predicts the return tomorrow is the AVG return until today -> passed
    XGBoost model is better than avg rmse AVG rolling window:   
    AVG RMSE: 0.017893688794979667 < 0.0292821198101994 -> 38% improvement
    
4. Linear regression -> better than the XGBoost WTF!!!!!!!!!!!!!!!!!!!
            AVG RMSE: 0.017780925353968597 < 0.017893688794979667
            almost the same, check other tests to see which is better


We will see if the RMSE of the XGBoost tree is smaller. (smaller == better).
"""

################################### compare to other models on right direction guess ###################################
"""
0. better then the Dumb Model 

2. better then the previous_daily_return:
   XGBoost model is better than previous_daily_return:
   AVG right direction guess: 74.09274193548387 > 49.41532258064516

3. better then the AVG_rolling_window:
   XGBoost model is better than AVG_rolling_window:
   AVG right direction guess: 74.09274193548387 > 49.435483870967744

4. WORST then the Linear regression 
    Linear_Regression is better than XGBoost model:
    AVG right direction guess: 80.46370967741936 < 74.09274193548387
"""
# I think we can delete all the functions in the '...'
# def previous_daily_return(train_set: pd.DataFrame, validation_set: pd.DataFrame) -> np.ndarray:
#     """
#     Predict each stock's next-day return as its previous daily return
#     from the last date in the train set.
#     """
#
#     # copy to not change the df itself
#     train_set = train_set.copy()
#     validation_set = validation_set.copy()
#
#     # make sure the data format is right
#     train_set["date"] = pd.to_datetime(train_set["date"])
#     validation_set["date"] = pd.to_datetime(validation_set["date"])
#
#     # max date of the train set is the date we take the daily return from
#     last_train_date = train_set["date"].max()
#
#     # taking the daily return
#     last_day_returns = (
#         train_set[train_set["date"] == last_train_date]
#         .set_index("symbol")["daily_return_percentage"]
#     )
#
#     y_pred = validation_set["symbol"].map(last_day_returns).to_numpy()
#
#     return y_pred


# def rolling_avg_baseline(
#     train_set: pd.DataFrame,
#     validation_set: pd.DataFrame,
#     num_of_days: int,
#     relevant_col_name: str = "daily_return_percentage"
# ) -> np.ndarray:
#    """
#     Return a np.ndarray of avg daily_return
#    :param train_set: The training set
#    :param validation_set: The validation set
#    :param num_of_days: The number of days to calculate the AVG on
#    :param relevant_col_name: The name of the col we calculate the AVG on
#    :return: np.ndarray of avg daily_return
#    """
#    train_set = train_set.copy()
#    validation_set = validation_set.copy()
#
#    train_set['date'] = pd.to_datetime(train_set['date'])
#    validation_set['date'] = pd.to_datetime(validation_set['date'])
#
#    train_set = train_set.sort_values(["symbol", "date"]).reset_index(drop=True)
#
#     # check that the col name is in the df
#    if (relevant_col_name not in train_set.columns):
#        raise ValueError(f'{relevant_col_name} is not a column in the df!')
#
#    # rolling average per symbol, using the last num_of_days known values
#    train_set["rolling_avg_pred"] = (
#        train_set.groupby("symbol")[relevant_col_name]
#        .transform(lambda s: s.rolling(window=num_of_days, min_periods=num_of_days).mean())
#    )
#
#    last_train_date = train_set["date"].max()
#
#    last_rolling_avg = (
#        train_set[train_set["date"] == last_train_date]
#        .set_index("symbol")["rolling_avg_pred"]
#    )
#
#    y_pred = validation_set["symbol"].map(last_rolling_avg).to_numpy()
#
#    return y_pred



def avg_without_x_values(values: list[float], x: float) -> float:
    """
    Return the average of all values except those equal to x.
    :param values: the list of values
    :param x: The value we ignore
    :return: The avg of the values without the x values.
    """
    values_without_x = [value for value in values if value != x]
    if not values_without_x:
        return 0.0
    return float(sum(values_without_x) / len(values_without_x))

# todo: make the threshold dynamic, UCT vibe, here or in the model_Expending_window_eval func
def right_direction_above_threshold_validation(
        y_pred: np.ndarray, y_validation: pd.Series | np.ndarray,
        initial_threshold: float, values_list: list[int]) -> list[int]:
    """
    Returns a list with 1/0/-1 of the entries, right/no guess/wrong
    :param y_pred: The predicted values
    :param y_validation: The real values
    :param initial_threshold: The initial_threshold we want to pass to even make a guess
                                initial_threshold in R+
    :param values_list: The list with the values
    :return: A list where entry is: 0 -> if we didn't guess because we didn't pass the threshold
                                    1-> if we were right
                                   -1 -> if we were wrong
    """
    if (initial_threshold < 0):
        raise ValueError("Initial threshold must be greater than or equal to zero.")

    for prediction, y_validation_value in zip(y_pred, y_validation):
        right_guess_flag = -1
        # checking the ABS value but appending a +/- flag
        if abs(prediction) > initial_threshold:
            # this return 1 iff prediction > 0 O.W -1
            # prediction != 0
            # check if they have the same direction
            if np.sign(prediction) == np.sign(y_validation_value):
                right_guess_flag = 1
            else:
                right_guess_flag = 0

        values_list.append(right_guess_flag)

    return values_list


# def the class object that the function receives
# to keep the code neat

class StockInfo(TypedDict):
    name: str
    position: int
    price: float
    next_day_price: float
    y_pred: float

def calculate_stocks_value( stocks_dict: Dict[str, StockInfo], flag: int | None = None) -> float:
    """
    Calculates the value in USD of the stocks we have.
    We use this to calculate the value today, and the value tomorrow.
    :param stocks_dict: A dict of the stocks and the position we have at each one
                    structure: { name, position, price, y_pred }
    :param flag: flag == 1 -> price today
                 flag == 2 -> Tomorrow's price
    :return: The value of the stocks we have in hand
    """

    value_in_USD = 0
    for stock in stocks_dict.values():
        how_many_stocks = stock.get('position')
        stock_price = 0
        # price today
        if ((flag == 1) or (flag is None)):
            stock_price = stock.get('price')
        # price tomorrow
        elif ( flag == 2):
            stock_price = stock.get('next_day_price')
        else:
            raise ValueError("flag must be either 1,2 or None")
        value_in_USD += how_many_stocks * stock_price

    return value_in_USD




# evaluation how much do the models make.
# todo: Think if should also return a dict of all the deals it did.
# todo: the fields can be dynamic
# todo: understand if buy is on 'close', and sell is on 'open' or what
def model_simulation(
    current_positions: list[int],
    y_pred: np.ndarray,
    stocks_prices: pd.Series | np.ndarray,
    next_day_prices: pd.Series | np.ndarray,
    stocks_names: list[str],
    action_strategy: int | None,
    number_of_buys: int | None = 0,
    number_of_sells: int | None = 0,
    min_y_pred_to_buy: float = 0.0,
    max_y_pred_to_sell: float = 0.0,
    money_in_our_possession_now: float = 0.0,
    transaction_fee: float = 0.0,
    max_spending_for_a_day: float = float("inf"),
) -> tuple[dict[str, Any], float, float, float, int, int]:
    """
    Calculate the delta of the portfolio.
    each iteration receives the new y_pred, new_stocks_prices, new money_in_our_possession_now
    for now the rest is
    :param current_positions: A list of the current positions
                              if we have x stocks of stock_0, we will have x in the
                              0 entry of the list.
    :param y_pred: The predicted values of a stock (next_day_return, close,...)
    :param stocks_prices: The prices of the stocks on current date, (I think 'open' is better the 'close')
    :param next_day_prices: The price of the stocks on the next date.
    :param stocks_names: The names of the stocks.
    :param action_strategy: A flag of what strategy to use todo: see how to use this
                            action_strategy == 1 -> Buy if: have mo
    :param number_of_buys: The amount of times we bought a stock
    :param number_of_sells: The amount of times we sold a stock
    :param min_y_pred_to_buy: We buy a stock only if its y_pred is over/equal this threshold
    :param max_y_pred_to_sell: We sell a stock only if its y_pred is under/equal this threshold
    :param money_in_our_possession_now: The amount of USD we have at a given moment ->R+
    :param transaction_fee: The cost of an action (maybe depends on amount of stock)
                            look luke it varies between stocks, fee * |stocks_bought|
                                todo: understand the transaction fee
    :param max_spending_for_a_day: The max amount we allow the model to spend in a day
    :return: New positions list, money_in_our_possession_now, money made/ lost
    """
    # todo: think of better strategy, for now it is this:
    # todo: Should always buy the stocks with the best predicted return
    # todo: No split, but only the stock with the best predicted return
    # todo: if can't buy more, move to the next one
    # todo: for now, there is no limit to the amount of money it can spend in a day
    # todo: sell the stocks with the worst predicted return that we have


    starting_money_amount = float(money_in_our_possession_now)
    # the cost of a stock is the close now + fees
    # we can buy a stock
    length_current_positions = len(current_positions)
    length_stocks_price = len(stocks_prices)
    length_next_day_stock_price = len(next_day_prices)
    length_y_pred = len(y_pred)
    length_stocks_names = len(stocks_names)
    if not (
            length_current_positions == length_stocks_price == length_next_day_stock_price
            == length_y_pred == length_stocks_names):
                raise ValueError("Current positions, stocks prices and y_pred don't match in length")

    if (money_in_our_possession_now <= 0):
        raise ValueError('what are you in debut? money_in_our_possession_now must be positive')

    if (transaction_fee < 0):
        raise ValueError('transaction_fee must be 0 or more.')

    # can't sort t_pred by value because I MUST keep the order between the 3 lists
    # can make a dict and then work -> todo is it the fastest right way?????
    og_dict_of_stocks = {}
    for stock_name, position, stock_price, stock_next_day_price, y_pred_stock in zip(stocks_names, current_positions,
                                                               stocks_prices, next_day_prices, y_pred):
        # We define the inner dictionary here so it exists before we try to access its fields
        og_dict_of_stocks[stock_name] = {
            'name': stock_name,
            'position': position,
            'price': float(stock_price),
            'next_day_price': float(stock_next_day_price),
            'y_pred': float(y_pred_stock)
        }

    value_of_stocks_in_hand_in_the_beginning = calculate_stocks_value(og_dict_of_stocks,1)
    # sorting the dict based on the y_pred value, desc

    dict_of_stocks = dict(
        sorted(
            og_dict_of_stocks.items(),
            key = lambda item: item[1]['y_pred'], #[0] is the key, we want the values, so [1]
            reverse = True
        )
    )

    # Explicitly casting the comparison values ensures the type checker is happy
    potentially_buy_stocks = {k: v for k, v in dict_of_stocks.items()
                              if float(v.get('y_pred', 0.0)) >= float(min_y_pred_to_buy)}

    potentially_sell_stocks = {k: v for k, v in dict_of_stocks.items()
                               if v.get('position', 0) != 0
                               and float(v.get('y_pred', 0.0)) <= float(max_y_pred_to_sell)}

    # SELLING BEFORE BUYING -> to make room
    # selling -> All of them have a worst/equal y_pred than the min -> sell all
    for stock in potentially_sell_stocks.values():
        stock_name = stock.get('name')
        # keeping track on the amount of stocks we sold
        number_of_sells += stock["position"]
        # based on the value of the stocks today and the money we have in hand
        money_in_our_possession_now += stock["position"] * (stock["price"] - transaction_fee)
        stock["position"] = 0


        # key -> name, update the position field to 0 to all these stocks
        og_dict_of_stocks[stock_name]['position'] = 0

    # buying -> according to budget + transaction_fee + max_spending_for_a_day
    spend = 0.0


    # dict is ordered by y_pred value, desc
    # we already checked we 'should' buy

    for stock in potentially_buy_stocks.values():
        stock_name = stock.get('name')
        stock_price = float(stock['price'])

        # We define these two helper variables right here inside the loop:
        price_with_fee = stock_price + transaction_fee

        # Defining the constraints based on your budget and the daily cap
        max_by_budget = money_in_our_possession_now // price_with_fee
        max_by_daily_limit = (max_spending_for_a_day - spend) // price_with_fee

        num_shares_to_buy = int(max(0.0, min(float(max_by_budget), float(max_by_daily_limit))))

        if num_shares_to_buy > 0:
            cost_of_transaction = num_shares_to_buy * price_with_fee
            stock['position'] += num_shares_to_buy
            spend += cost_of_transaction
            money_in_our_possession_now -= cost_of_transaction
            # keeping track on the amount of stocks we bought
            number_of_buys += num_shares_to_buy

            # Update the original dict record
            og_dict_of_stocks[stock_name]['position'] = stock['position']

    # the value of the stocks we have in hand after selling and buying
    # running the function on the og_dict_of_stocks because we updated it
    stock_in_hand_value_next_day = calculate_stocks_value(og_dict_of_stocks,2)

    # money_made_lost = Total Portfolio Value - starting_money_amount - value of stocks in hand

    portfolio_value_next_day = float(money_in_our_possession_now + stock_in_hand_value_next_day)

    # todo: fix this and run it in the model_evl function
    money_made_lost = (portfolio_value_next_day
                       - starting_money_amount - value_of_stocks_in_hand_in_the_beginning)

    # og_dict_of_stocks -> is the positions list of the stocks
    # money_in_our_possession_now -> USD in hand
    # money_made_lost -> delta of the portfolio value
    return (og_dict_of_stocks, float(money_in_our_possession_now), float(money_made_lost),
            portfolio_value_next_day, number_of_buys, number_of_sells)



# this function is a wrapper function that runs the simulation
# The only model that need the num_of_days field is the RollingAvgModel,
# and he receives it in the __init__.
def run_model_simulation_backvalidation(
    df: pd.DataFrame,
    feature_cols_x: list[str],
    target_col: str,
    start_date_train_window: date,
    end_date_train_window: date,
    model: Model, # <--- The Model Object
    price_col: str = "close",
    initial_cash: float = 10_000.0,
    min_y_pred_to_buy: float = 0.0,
    max_y_pred_to_sell: float = 0.0,
    transaction_fee: float = 0.0,
    max_spending_for_a_day: float = float("inf"),
) -> tuple[float, float, pd.DataFrame, int, int]:
    """
    Run an expanding-window test using model_simulation().

    :param df:
    :param feature_cols_x:
    :param target_col:
    :param start_date_train_window:
    :param end_date_train_window:
    :param model:
    :param price_col:
    :param initial_cash: The cash in USD we start with.
    :param min_y_pred_to_buy:
    :param max_y_pred_to_sell:
    :param transaction_fee: The transaction fee per stock.
    :param max_spending_for_a_day: A limit we decide to not go over.
    :return:
    - final money made/lost
    - final percent made/lost
    - history dataframe
    """

    if not set(feature_cols_x).issubset(df.columns):
        raise ValueError('all the feature_cols_x must be in the df.')
    # needs to be tolist()?
    if not(target_col in df.columns):
        raise ValueError("Target column must be in df.")

    if not (price_col in df.columns):
        raise ValueError("Price column must be in df.")

    if (transaction_fee < 0):
        raise ValueError("Transaction fee must be positive.")

    if (max_spending_for_a_day <= 0):
        raise ValueError("Max spending must be positive.")

    if (initial_cash <= 0):
        raise ValueError("Max spending must be positive.")


    (
        og_df,
        start_date_train_window,
        end_date_train_window,
        max_date,
        trading_dates,
        future_validation_dates,
        current_validation_date
    ) = data_prep(
        df=df,
        feature_cols_x=feature_cols_x,
        target_col=target_col,
        start_date_train_window=start_date_train_window,
        end_date_train_window=end_date_train_window
    )


    cash = float(initial_cash)
    positions_dict: dict[str, Any] | None = None
    history_rows: list[dict[str, Any]] = []
    number_of_buys = 0
    number_of_sells = 0

    # not all the stocks have the max_date because of the train_val, test split
    # some have one day less, so we do current_validation_date < max_date
    while current_validation_date <= max_date:
        next_dates = trading_dates[trading_dates > current_validation_date]
        if next_dates.empty:
            break

        next_trade_date = pd.to_datetime(next_dates.iloc[0]).normalize()

        train_set, validation_set = data_split_to_train_and_validation(
            og_df,
            start_date_train_window,
            current_validation_date
        )

        if train_set.empty or validation_set.empty:
            current_validation_date = next_trade_date
            continue

        x_train = train_set[feature_cols_x]
        y_train = train_set[target_col]
        x_validation = validation_set[feature_cols_x]
        y_validation = validation_set[target_col]

        # OOP -> the model is a var we get.
        # all the models that have more fields in the fit/predict function, they receive them
        # in the creation of the model itself, so it is fine you don't see them here

        # the x_train, y_train are only in the [Start date of the window,end date of the window] days,
        # so we can, and should scale the x_train, for the relevant models, and it won't create a leakage.
        # same logic for the validation set for the reset of the dates.
        # scaling doesn't really hurt a Model, and since we haven't got a flag here, and it is not to
        # time-consuming, we just scale for every model on the dates of the window.
        scaler = MinMaxScaler()
        x_train_scaled = pd.DataFrame(
            scaler.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index
        )
        x_validation_scaled = pd.DataFrame(
            scaler.transform(x_validation),
            columns=x_validation.columns,
            index=x_validation.index
        )

        # we don't scale the y/target col.
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(
            x_test=x_validation_scaled,
            train_set=train_set,
            val_set=validation_set
        )


        stock_names = validation_set["symbol"].tolist()
        current_positions = positions_dict_to_list(positions_dict, stock_names)

        today_prices = validation_set[price_col].to_numpy(dtype=float)

        next_day_df = og_df.loc[
            og_df["date"] == next_trade_date,
            ["symbol", price_col]
        ].copy()

        next_day_price_map = dict(zip(next_day_df["symbol"], next_day_df[price_col]))

        missing_symbols = [name for name in stock_names if name not in next_day_price_map]
        if missing_symbols:

            raise ValueError(f"Missing next-day prices for symbols: {missing_symbols} " # 2025-05-19 for <= and for <
                             f"in date: {current_validation_date}") #

        next_day_prices = np.array(
            [float(next_day_price_map[name]) for name in stock_names],
            dtype=float
        )

        (positions_dict, cash, money_made_lost_step, portfolio_value_next_day,
         number_of_buys, number_of_sells) = model_simulation(
            current_positions=current_positions,
            y_pred=y_pred,
            stocks_prices=today_prices,
            next_day_prices=next_day_prices,
            stocks_names=stock_names,
            action_strategy=1,
            number_of_buys = number_of_buys,
            number_of_sells = number_of_sells,
            min_y_pred_to_buy=min_y_pred_to_buy,
            max_y_pred_to_sell=max_y_pred_to_sell,
            money_in_our_possession_now=cash,
            transaction_fee=transaction_fee,
            max_spending_for_a_day=max_spending_for_a_day
        )

        percent_made_lost_step = (
            (portfolio_value_next_day - initial_cash) / initial_cash * 100
            if initial_cash != 0 else 0.0
        )

        history_rows.append({
            "validation_date": current_validation_date,
            "next_trade_date": next_trade_date,
            "cash": float(cash),
            "portfolio_value_next_day": float(portfolio_value_next_day),
            "money_made_lost_step": float(money_made_lost_step),
            "percent_made_lost_total": float(percent_made_lost_step),
        })

        current_validation_date = next_trade_date

    history_df = pd.DataFrame(history_rows)

    if history_df.empty:
        return 0.0, 0.0, history_df, number_of_buys, number_of_sells

    final_portfolio_value = float(history_df["portfolio_value_next_day"].iloc[-1])
    final_money_made_lost = final_portfolio_value - float(initial_cash)
    final_percent_made_lost = (
        (final_money_made_lost / float(initial_cash)) * 100
        if initial_cash != 0 else 0.0
    )

    return final_money_made_lost, final_percent_made_lost, history_df, number_of_buys, number_of_sells




############################## try to split the function ##############################
def y_pred_based_on_model_object(
        model: Model,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_validation: pd.DataFrame,
        train_set: pd.DataFrame,
        validation_set: pd.DataFrame
) -> tuple[np.ndarray, Model]:
    """
    Standardized wrapper for any TradingModel object.
    OOP -> the model is a var we get.
    all the models that have more fields in the fit/predict function, they receive them
    in the creation of the model itself, so it is fine you don't see them here
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(
        x_test=x_validation,
        train_set=train_set,
        val_set=validation_set
    )


    return np.asarray(y_pred), model


def validation_result_based_on_test_type(
    validation_type: int | None,
    values_list: list[float | int],
    y_validation: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    initial_threshold: float | None = None,
) -> list[float | int]:
    """
    This returns the values_list with the appended new value for the given set and test type.
    :param validation_type: The type of test.
            validation_type is None or 1 -> RMSE, validation_type == 2 -> right direction,
            validation_type == 3 -> right direction over threshold
    :param values_list: A list of the values of the validation.
    :param y_validation: the Y_test values.
    :param y_pred: y_pred values.
    :param initial_threshold: the initial threshold, make action only if the y_pred of
                              a stock is over the threshold.
    :return: The values_list after adding the new value of the last validation.
    """
    # RMSE
    if ((validation_type is None) or (validation_type == 1) ):
        values_list.append(root_mean_squared_error(y_validation, y_pred))

    # right direction
    elif (validation_type == 2):
        value_to_append = (y_pred * y_validation > 0).mean()
        values_list.append(value_to_append)

    # right direction over threshold
    elif (validation_type == 3):
        if initial_threshold is None:
            raise ValueError('for validation type 3 -> right direction predict when change is significant,'
                             ' an initial threshold is must!!!')
        else:
            # num_above_threshold = np.sum(np.abs(y_pred) > initial_threshold)
            # print("num_above_threshold:", num_above_threshold, "out of", len(y_pred))
            values_list = right_direction_above_threshold_validation(y_pred, y_validation, initial_threshold, values_list)

    else:
        raise ValueError('validation_type must be 2,3 or None')

    return values_list

# validation_type == 3 -> right direction for |y_pred| >= threshold
def validation_type_3_helper(values_list: list[int]) -> tuple[float, int, int]:
    """
    Return the avg_value_in_list, num_of_right_decisions, num_of_wrong_decisions
    :param values_list: The values list of the validations.
    :return: avg_value_in_list, num_of_right_decisions, num_of_wrong_decisions
    """
    # -1 is a flag we did reach the threshold, so we didn't guess
    # 0 -> wrong guess
    # 1 -> right guess
    avg_value_in_list = avg_without_x_values(values_list, -1)
    # count the number of times we made a decision
    # #1 + # 0
    num_of_right_decisions = sum([1 for num in values_list if num == 1])
    num_of_wrong_decisions = sum([1 for num in values_list if num == 0])
    return avg_value_in_list, num_of_right_decisions, num_of_wrong_decisions


def positions_dict_to_list(
    positions_dict: dict[str, Any] | None,
    stock_names: list[str]
) -> list[int]:
    """
    Convert the positions dict into a list aligned with stock_names order.
    """
    if positions_dict is None:
        return [0] * len(stock_names)

    return [
        int(positions_dict.get(name, {}).get("position", 0))
        for name in stock_names
    ]

def data_prep(
    df: pd.DataFrame,
    feature_cols_x: list[str],
    target_col: str,
    start_date_train_window: date,
    end_date_train_window: date
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Series, pd.Series, pd.Timestamp]:
    """
    This function prepare the dateץ
    1. Drops rows with NA values in important fields.
    2. Takes only the data of the dates in [start_date_train_window,end_date_train_window]
    :param df: The df.
    :param feature_cols_x: The feature_cols_x.
    :param target_col: The y_col.
    :param start_date_train_window: start of the rolling window.
    :param end_date_train_window: end of the rolling. window.
    :return: The vars after cleaning them.
    """
    # The validation set is just the records of the date after the closing of the window.
    # So there is no need to a 'validation_set'.

    if not set(feature_cols_x).issubset(set(df.columns)):
        raise ValueError('feature_cols_x must be in the df!')

    og_df = df.copy()
    og_df = og_df.dropna(
        subset=feature_cols_x + [target_col]).copy()  # drop all rows with an NA value in the important fields for train
    og_df["date"] = pd.to_datetime(og_df["date"]).dt.normalize()
    start_date_train_window = pd.to_datetime(start_date_train_window).normalize()
    end_date_train_window = pd.to_datetime(end_date_train_window).normalize()
    max_date = og_df["date"].max()
    max_date = pd.to_datetime(max_date).normalize()

    current_end_date = end_date_train_window
    # loop on different window sizes and calculate the RMSE foreach
    trading_dates = (
        og_df["date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if (target_col not in og_df.columns):
        raise ValueError('target_col must be in og_df.columns')

    future_validation_dates = trading_dates[trading_dates > end_date_train_window]
    if future_validation_dates.empty:
        print(f'\n this is the end_date_train_window: {end_date_train_window}')
        raise ValueError("No validation dates available after the initial train window.")

    current_validation_date = pd.to_datetime(future_validation_dates.iloc[0]).normalize()

    # print("target min/max:", og_df[target_col].min(), og_df[target_col].max())
    # print("threshold:", initial_threshold)

    return (og_df, start_date_train_window, end_date_train_window, max_date,
            trading_dates, future_validation_dates, current_validation_date)





# Expending window idea to evaluate the model
# it is for evaluation only, one do not create one model
# from all the ones you created.
# it helps creating a better model only because it
# helps when selecting the hyper paramaters.
# todo: does it need to return the trained model or just train it?
# todo: read what chat wrote.
# todo: structure this function to many smaller ones, way to big
def model_Expending_window_eval(
    df: pd.DataFrame,
    feature_cols_x: list[str],
    target_col: str,
    start_date_train_window: date,
    end_date_train_window: date,
    flag: int,
    num_of_days: int | None = None,
    validation_type: int | None = None,
    initial_threshold: float | None = None,
) -> float | tuple[float, int, int]:
    """
    Train the model on [start_date_train_window,end_date_train_window]
    test it on [end_date_train_window + 1]
    Calculate the lost between the predicted value to the real value
    Append it to a list
    Return the AVG lost of the list
    (All stocks have the same max date and all stocks have records to all trading dates until max date)

    NOTICE: we call this function only inside other functions, and we do the error handing inside
            the other functions -> so we didn't do it here.

    NOTICE: WE SCALE THE X_FEATURES HERE!!!!
            Not all models need the x_features to be scaled, (XGBoost), so we scale where needed.
            We scale using Min-Max scaler, WE SCALE ONLY ON THE [Start date of the window,end date of the window]
            WINDOW, if we scale on all the data it will be data leakage.
            flag == 0 -> dumb model: no scaling ,
            flag == 1 ->XGBoost model: no scaling
            flag == 2 -> previous daily return pred: no scaling,
            flag == 3 -> rolling AVG pred: no scaling
            flag == 4 -> Linear regression: scaling
            flag == 5 -> FullyConnectedNeuralNetwork: scaling

    :param df: The whole df
    :param feature_cols_x: Feature columns for the X
    :param target_col: Feature columns for the Y
    :param start_date_train_window: Start date of the window
    :param end_date_train_window: end date of the window
    :param flag: the flag is to see if we work on which model we want to work on:
                 flag == 0 -> dumb model,
                 flag == 1 ->XGBoost model,
                 flag == 2 -> previous daily return pred, flag == 3 -> rolling AVG pred
                 flag == 4 -> Linear regression
                 flag == 5 -> FullyConnectedNeuralNetwork
    :param num_of_days: The number of days to calculate the AVG on
    :param validation_type: Type of test we want to do
                    validation_type == 1 or is None -> AVG RMSE
                    validation_type == 2 -> right direction: y_prd * t_validation > 0 <-> 1 O.W 0
                    NOT EQUAL TO 0 BECAUSE FOR THE DUMB MODEL y_prd = 0 SO y_prd * t_validation = 0 ALWAYS
                    validation_type == 3 -> make guess only if |y_pred| > initial_threshold
                    validation_type == 4 -> money made
    :param initial_threshold: This is the initial_threshold the |y_pred| should be over to make a guess
                              So 0.02 is 2%  for example if target_col is next_day_return
    :return: The AVG metrik of the list or, if validation 3 -> AVG metrika, num of right decisions, num of wrong decisions
    """

    # todo: i want to add here validation_type == 4 -> money made

    (og_df, start_date_train_window, end_date_train_window,
     max_date, trading_dates,
     future_validation_dates, current_validation_date) = data_prep(df = df, feature_cols_x = feature_cols_x,
                                                       target_col = target_col,
                                                       start_date_train_window = start_date_train_window,
                                                       end_date_train_window = end_date_train_window)
    
    values_list = []

    while current_validation_date <= max_date:
        train_set, validation_set = data_split_to_train_and_validation(og_df,
                                                                       start_date_train_window,
                                                                       current_validation_date)

        if validation_set.empty:
            break


        x_train = train_set[feature_cols_x]
        y_train = train_set[target_col]

        x_validation = validation_set[feature_cols_x]
        y_validation = validation_set[target_col]

        # this is like dumb model -> flag == 0
        relevant_col_name = 'daily_return_percentage'

        num_of_x_fields = len(feature_cols_x)
        # creating the model -> OOP
        # todo: does this work for the ML models?
        model = get_model(flag=flag, num_of_days=num_of_days,
                          relevant_col_name=relevant_col_name, num_of_x_fields = num_of_x_fields )

        # the x_train, y_train are only in the [Start date of the window,end date of the window] days
        # so we can, and should scale the x_train, for the relevant models, and it won't create a leakage.
        # same logic for the validation set for the reset of the dates.
        if(flag in relevant_models_and_numbers_dict_for_scaling):
            scaler = MinMaxScaler()
            x_train_scaled = pd.DataFrame(
                scaler.fit_transform(x_train),
                columns=x_train.columns,
                index=x_train.index
            )
            x_validation_scaled = pd.DataFrame(
                scaler.transform(x_validation),
                columns=x_validation.columns,
                index=x_validation.index
            )

            # we don't scale the y/target col.
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(
                x_test=x_validation_scaled,
                train_set=train_set,
                val_set=validation_set
            )
        else:
            model.fit(x_train, y_train)
            y_pred = model.predict(
                x_test=x_validation,
                train_set=train_set,
                val_set=validation_set
            )


        ##################### validation type selection #####################
        values_list = validation_result_based_on_test_type(validation_type = validation_type, values_list = values_list,
                                                     y_validation = y_validation, y_pred = y_pred,
                                                     initial_threshold = initial_threshold)

        next_dates = trading_dates[trading_dates > current_validation_date]
        if next_dates.empty:
            break
        current_validation_date = pd.to_datetime(next_dates.iloc[0]).normalize()



    if not values_list:
        raise ValueError("No evaluation windows were created.")


    avg_value_in_list = float(np.mean(values_list))

    # validation_type == 1 -> RMSE, we don't multiply by 100
    if ((validation_type is None) or (validation_type == 1)):
        return avg_value_in_list

    # validation_type == 2 -> right direction
    elif (validation_type == 2):
        return avg_value_in_list * 100

    # validation_type == 3 -> right direction |y_pred| > initial_threshold
    elif(validation_type == 3):
        return validation_type_3_helper(values_list)


    # todo: fix this, make the money compression
    # for now, we do the money compression in a different place
    else:
        raise ValueError('The money compression is in a different place.')










# todo: fix direction of the > <
# todo: make sure there is now leakage
def compare_models(model_1_name: str, model_1_score: float,
                   model_2_name: str, model_2_score: float,
                   validation_type: int | None = None,
                   num_of_right_decisions_model_1: int| None = None, num_of_wrong_decisions_model_1: int| None = None,
                   num_of_right_decisions_model_2: int| None = None, num_of_wrong_decisions_model_2: int| None = None,):
    # Determine the label
    """
    This just compares the output of two different models on the same validation.
    :param model_1_name: The name of the first model.
    :param model_1_score: The score of the first model on the selected test type.
    :param model_2_name: The name of the second model.
    :param model_2_score: The score of the second model on the selected test type.
    :param validation_type: the type of test we want to do
                            validation_type == 1 or is None -> AVG RMSE
                            validation_type == 2 -> AVG % right direction
                            validation_type == 3 -> AVG % right direction over threshold
    :return: Nothing, just prints the comparison results.
    """
    # so validation_type is None or validation_type == 1 ->
    # we do the AVG RMSE test.
    text = 'AVG RMSE'
    if(validation_type == 2):
        text = 'AVG % right direction'
    elif(validation_type == 3):
        text = 'AVG % right direction over threshold'

    # Logic:
    # If type 1: model_1 is better if score is LOWER
    # If type 2: model_1 is better if score is HIGHER
    # if type 3: model_1 is better if score is HIGHER
    if ((validation_type is None) or (validation_type == 1)):
        m1_wins = model_1_score < model_2_score
        operator = "<"
    else:
        m1_wins = model_1_score > model_2_score
        operator = ">"

    if m1_wins:
        print(f"{model_1_name} is better than {model_2_name}:")
        print(f"{text}: {model_1_score} {operator} {model_2_score}\n")
    else:
        print(f"{model_2_name} is better than {model_1_name}:")
        # For the "else", we flip the operator for the printout
        alt_operator = ">" if operator == "<" else "<"
        print(f"{text}: {model_2_score} {alt_operator} {model_1_score}\n")

    if validation_type == 3:
        print(f'This is the right direction over threshold comp:\n')
        print(f'Num of right decisions {model_1_name} : {num_of_right_decisions_model_1},'
              f'Num of wrong decisions {model_1_name} : {num_of_wrong_decisions_model_1}')

        print(f'Num of right decisions model {model_2_name} : {num_of_right_decisions_model_2},'
              f'Num of wrong decisions model {model_2_name} : {num_of_wrong_decisions_model_2}\n')




def money_made_comparison(name_model_1,money_made_lost_model_1, percent_made_lost_1, number_of_buys_model_1,
                          number_of_sells_model_1, history_df_model_1,
                          name_model_2,money_made_lost_model_2, percent_made_lost_2, number_of_buys_model_2,
                          number_of_sells_model_2, history_df_model_2
                          ) -> None:
    """
    Compares two models, which made more money in a given time period and budget.
    :param name_model_1: The name of model 1
    :param percent_made_lost_1: the % model 1 made/ lost
    :param number_of_buys_model_1: The number of stocks model 1 bought
    :param number_of_sells_model_1: The number of stocks model 1 sold
    :param history_df_model_1: the df of the buys/ sells of model 1
    :param name_model_2: The name of model 2
    :param percent_made_lost_2: the % model 2 made/ lost
    :param number_of_buys_model_2: The number of stocks model 2 bought
    :param number_of_sells_model_2: The number of stocks model 2 sold
    :param history_df_model_2: the df of the buys/ sells of model 2
    :return: Nothing.
    """

    print(f"Money made/lost {name_model_1}:", money_made_lost_model_1)
    print(f"Percent made/lost: {name_model_1}", percent_made_lost_1)
    print(f'amount of stocks bought {name_model_1}: {number_of_buys_model_1}')
    print(f'amount of stocks sold {name_model_1}: {number_of_sells_model_1}')
    print(history_df_model_1.tail())
    print(f'number of trading days for both models:'
          f'{name_model_1}: {len(history_df_model_1)} ,'
          f'{name_model_2}: {len(history_df_model_2)}')

    print(f"Money made/lost {name_model_2}:", money_made_lost_model_2)
    print(f"Percent made/lost: {name_model_2}", percent_made_lost_2)
    print(f'amount of stocks bought {name_model_2}: {number_of_buys_model_2}')
    print(f'amount of stocks sold {name_model_2}: {number_of_sells_model_2}')
    print(history_df_model_2.tail())
    print(f'number of trading days {name_model_2}: {len(history_df_model_2)}')

    if (money_made_lost_model_1 > money_made_lost_model_2):
        print(f'{name_model_1} made more money.')
    else:
        print(f'{name_model_2} made more money.')

    print("-----------------------------------------------------\n")




# todo: find how to use dates because I understand XGBoost can't use dates
# todo: find using a greedy approach what are the best comb of
# todo: fields is the best
# todo: think of a way to calculate the regret -> write it in the eval.py file
# todo: think about adding Heuristics


def all_dict_keys_and_values(options_dict:  dict[int, str] | None = None):
    """
    Returns a string with the names and numbers of all the models.
    Uses a gloabal dict of the models and their names if nothing else is given.
    If a dict is given, we just work on it.
    :param options_dict: The dict we want to loop over and print.
    :return: A string with the names and numbers of all the models.
    """

    if options_dict is None:
        options_dict = relevant_models_and_numbers_dict

    result = "\nThese are the options: \n"
    for option in options_dict:
        result += str(option) + ": " + options_dict[option] + "\n"
    result += '\nenter the number you want'

    return result

def model_score_getter(model_flag: int, test_type: int | None,
                       num_of_days: int| None = 5,
                       initial_threshold: float| None = 0.015) -> float | tuple[float, int, int]:
    """
    Gets a model_flag and wanted test type, returns the score of the model
    :param model_flag: The type of model we want to create
            model_flag == 0 -> dumb model
            model_flag == 1 -> XGBoost model
            model_flag == 2 -> previous daily return pred
            model_flag == 3 -> rolling AVG pred
            model_flag == 4 -> Linear regression
            model_flag == 5 -> FullyConnectedNeuralNetwork

    :param test_type: The type of test we want to do.
            test_type = 1 -> RMSE
            test_type = 2 -> right direction
            test_type = 3 -> right direction over threshold


    :param num_of_days: The num of days to use for the rolling avg score -> relevant only for this.
    :param initial_threshold: The initial threshold -> relevant only if we do the right direction over threshold test.
    :return: The score of the selected model on the selected test type.
    """


    # todo: relevant_models_and_numbers_dict is hard coded here -> maybe change later
    if not (model_flag in relevant_models_and_numbers_dict):
        raise ValueError('model_flag must be in the relevant_models_and_numbers_dict.')

    if not((test_type is None) or (test_type in validation_types_dict)):
        raise ValueError('test_flag is not valid.')

    # we will use these var only for test_type == 3
    avg_right_direction_above_threshold_guess_model = 0.0
    num_of_right_decisions_direction_above_threshold_model = 0
    num_of_wrong_decisions_direction_above_threshold_model = 0

    model_score = None
    # RMSE or right direction
    if ((test_type is None) or (test_type == 1) or (test_type == 2)):
        if(model_flag == 3):
            model_score = model_Expending_window_eval(
                df=data_experiment_train_and_validation_df,
                feature_cols_x=feature_cols_x,
                target_col='next_day_return',
                start_date_train_window=initial_train_start,
                end_date_train_window=initial_train_end,
                flag=model_flag,
                num_of_days = num_of_days,
                validation_type = test_type
            )
        else:
            model_score = model_Expending_window_eval(
                df=data_experiment_train_and_validation_df,
                feature_cols_x=feature_cols_x,
                target_col='next_day_return',
                start_date_train_window=initial_train_start,
                end_date_train_window=initial_train_end,
                flag=model_flag,
                validation_type=test_type
            )

        return model_score


    # right direction over threshold
    elif(test_type == 3):
        # rolling AVG pred
        if(model_flag == 3):
            (avg_right_direction_above_threshold_guess_model,
             num_of_right_decisions_direction_above_threshold_model,
             num_of_wrong_decisions_direction_above_threshold_model)\
                = model_Expending_window_eval(
                df=data_experiment_train_and_validation_df,
                feature_cols_x=feature_cols_x,
                target_col='next_day_return',
                start_date_train_window=initial_train_start,
                end_date_train_window=initial_train_end,
                flag = model_flag,
                num_of_days = num_of_days,
                validation_type = test_type,
                initial_threshold = initial_threshold
            )

        # for every model != 3 -> there is no need for the num_of_days = num_of_days var,
        # that is the only difference.
        else:
            (avg_right_direction_above_threshold_guess_model,
             num_of_right_decisions_direction_above_threshold_model,
             num_of_wrong_decisions_direction_above_threshold_model) \
                = model_Expending_window_eval(
                df=data_experiment_train_and_validation_df,
                feature_cols_x=feature_cols_x,
                target_col='next_day_return',
                start_date_train_window=initial_train_start,
                end_date_train_window=initial_train_end,
                flag=model_flag,

                validation_type=test_type,
                initial_threshold=initial_threshold
            )

    # NOTICE: we don't have to write the last test_type in an "elif", it can be in an "else"
    # because we already checked the test_type is legal.
    # I choose to write it as such because it is clearer!



    return (avg_right_direction_above_threshold_guess_model,
            num_of_right_decisions_direction_above_threshold_model,
            num_of_wrong_decisions_direction_above_threshold_model)






def test_type_models_to_compare():
    """
    This is a helper function to help as compare between two models.
    We select:
        1. test type -> RMSE, right direction, right_direction_over_threshold
        2. the model to compare between
    :return: prints the results of the comparison
    """
    model_name_1 = None
    model_name_2 = None
    while(True):
        print(all_dict_keys_and_values())
        model_flag_1 = int(input("Type the number of model 1: "))
        model_flag_2 = int(input("Type the number of model 2: "))
        if ( (model_flag_1 in relevant_models_and_numbers_dict) and
                (model_flag_2 in relevant_models_and_numbers_dict) and
                (model_flag_1 != model_flag_2) ):
            model_name_1 = relevant_models_and_numbers_dict[model_flag_1]
            model_name_2 = relevant_models_and_numbers_dict[model_flag_2]
            break
        else:
            print("Invalid choice.")


    # making sure the users write a valid answer
    while(True):
        print(all_dict_keys_and_values(validation_types_dict))
        test_flag = int(input('select the test type you want to use for the comparison: '))
        if test_flag in validation_types_dict:
            break
        else:
            print("Invalid choice.")


    if (test_flag == 3):
        (avg_right_direction_above_threshold_guess_model_1,
         num_of_right_decisions_direction_above_threshold_model_1,
         num_of_wrong_decisions_direction_above_threshold_model_1) = model_score_getter(model_flag = model_flag_1,
                                                                                   test_type = test_flag)

        (avg_right_direction_above_threshold_guess_model_2,
         num_of_right_decisions_direction_above_threshold_model_2,
         num_of_wrong_decisions_direction_above_threshold_model_2) = model_score_getter(model_flag = model_flag_2,
                                                                                   test_type = test_flag)

        compare_models(model_1_name = model_name_1, model_1_score = avg_right_direction_above_threshold_guess_model_1,
                       model_2_name = model_name_2, model_2_score =  avg_right_direction_above_threshold_guess_model_2,
                       validation_type = test_flag,
                       num_of_right_decisions_model_1 = num_of_right_decisions_direction_above_threshold_model_1,
                       num_of_wrong_decisions_model_1 = num_of_wrong_decisions_direction_above_threshold_model_1,
                       num_of_right_decisions_model_2 = num_of_right_decisions_direction_above_threshold_model_2,
                       num_of_wrong_decisions_model_2 = num_of_wrong_decisions_direction_above_threshold_model_2)


    else:
        model_1_score = model_score_getter(model_flag = model_flag_1, test_type = test_flag)
        model_2_score = model_score_getter(model_flag = model_flag_2, test_type = test_flag)

        # comparing the models
        compare_models(model_1_name = model_name_1, model_1_score = model_1_score,
                       model_2_name = model_name_2, model_2_score = model_2_score,
                       validation_type = test_flag)





"""
Inside this block we write all the code that we want to execute each time we run the code.
"""
if __name__ == "__main__":
    data_experiment_train_and_validation_df = unpickle_data(experiment_train_and_validation_pickle_file_path)
    data_experiment_train_and_validation_df = data_experiment_train_and_validation_df.loc[:, ~data_experiment_train_and_validation_df.columns.duplicated()].copy()

    # a part to check there is no data leakage
    df_check = data_experiment_train_and_validation_df.copy()
    df_check["date"] = pd.to_datetime(df_check["date"])
    df_check = df_check.sort_values(["symbol", "date"]).reset_index(drop=True)

    # same-day return: (close_t - close_t-1) / close_t-1
    df_check["same_day_return_check"] = (
        df_check.groupby("symbol")["close"].pct_change(1)
    )

    # next-day return on row t: (close_t+1 - close_t) / close_t
    df_check["next_day_return_check"] = (
            df_check.groupby("symbol")["close"].shift(-1) / df_check["close"] - 1
    )

    print("Matches same-day target:",
          np.isclose(
              df_check["daily_return_percentage"],
              df_check["same_day_return_check"],
              equal_nan=True
          ).mean())

    print("Matches next-day target:",
          np.isclose(
              df_check["next_day_return"],
              df_check["next_day_return_check"],
              equal_nan=True
          ).mean())

    ## running the code
    initial_train_start = pd.Timestamp("2020-09-30").normalize()
    # initial_train_end = (pd.Timestamp("2026-03-16") - pd.Timedelta(days=15)).normalize()
    # the max date we have on the train_val_set right now is the max_date (2025-05-20 00:00:00)
    # so if we want 6 months before it ->
    initial_train_end = pd.Timestamp("2024-11-20").normalize()  # for now, hard code a date from about 6 months ago

    # todo: for now we didn't take the currency (all are USD), date fields as well -> need to find a way to use them
    feature_cols_x = (
        data_experiment_train_and_validation_df
        .drop(columns=['next_day_return', 'currency', 'exchange', 'symbol',
                       'date'])  # next_day_return is ratio (1 == 100%)
        .columns
        .tolist()
    )

    ############################################### running the validations ###############################################

    """
    I made a dynamic function that asks the user:
    1. what models he want to compare.
    2. in what type of test he wants to compare them.
    """
    test_type_models_to_compare()

    #######################################running the money made simulation#######################################

    """
    for now, the only models that need the relevant_col_name field are the PreviousDayReturnModel 
    and RollingAvgModel, and they receive it in the __init__.
    RollingAvgModel is the only model that need the num_of_days field, and he recives it in the __init__.
    """

    # XGBoost
    (money_made_lost_XGBoost,
     percent_made_lost_XGBoost,
     history_df_XGBoost, number_of_buys_XGBoost,
     number_of_sells_XGBoost) = run_model_simulation_backvalidation(
        df=data_experiment_train_and_validation_df,
        feature_cols_x=feature_cols_x,
        target_col="next_day_return",
        start_date_train_window=initial_train_start,
        end_date_train_window=initial_train_end,
        model=get_model(flag=1, relevant_col_name="daily_return_percentage"),  # XGBoost
        price_col="close",
        initial_cash=10_000.0,
        min_y_pred_to_buy=0.02,
        max_y_pred_to_sell=-0.02,
        transaction_fee=0.0,
        max_spending_for_a_day=3_000.0
    )

    # previous_daily_return
    # (money_made_lost_previous_daily_return,
    #  percent_made_lost_previous_daily_return,
    #  history_df_previous_daily_return, number_of_buys_previous_daily_return,
    #  number_of_sells_previous_daily_return) = run_model_simulation_backvalidation(
    #     df=data_experiment_train_and_validation_df,
    #     feature_cols_x=feature_cols_x,
    #     target_col="next_day_return",
    #     start_date_train_window=initial_train_start,
    #     end_date_train_window=initial_train_end,
    #     model=get_model(flag = 2, relevant_col_name = "daily_return_percentage"),  # previous_daily_return
    #     price_col="close",
    #     initial_cash=10_000.0,
    #     min_y_pred_to_buy=0.02,
    #     max_y_pred_to_sell=-0.02,
    #     transaction_fee=0.0,
    #     max_spending_for_a_day=3_000.0
    # )

    # rolling_avg_baseline
    # (money_made_lost_rolling_avg_baseline,
    #  percent_made_lost_rolling_avg_baseline,
    #  history_df_rolling_avg_baseline, number_of_buys_rolling_avg_baseline,
    #  number_of_sells_rolling_avg_baseline) = run_model_simulation_backvalidation(
    #     df=data_experiment_train_and_validation_df,
    #     feature_cols_x=feature_cols_x,
    #     target_col="next_day_return",
    #     start_date_train_window=initial_train_start,
    #     end_date_train_window=initial_train_end,
    #     model=get_model(flag = 3,num_of_days=5, relevant_col_name="daily_return_percentage"),  # rolling_avg_baseline
    #     price_col="close",
    #     initial_cash=10_000.0,
    #     min_y_pred_to_buy=0.02,
    #     max_y_pred_to_sell=-0.02,
    #     transaction_fee=0.0,
    #     max_spending_for_a_day=3_000.0
    # )

    # Linear_Regression
    (money_made_lost_Linear_Regression,
     percent_made_lost_Linear_Regression,
     history_df_Linear_Regression, number_of_buys_Linear_Regression,
     number_of_sells_Linear_Regression) = run_model_simulation_backvalidation(
        df=data_experiment_train_and_validation_df,
        feature_cols_x=feature_cols_x,
        target_col="next_day_return",
        start_date_train_window=initial_train_start,
        end_date_train_window=initial_train_end,
        model=get_model(flag = 4, relevant_col_name = "daily_return_percentage"),  # Linear Regression
        price_col="close",
        initial_cash=10_000.0,
        min_y_pred_to_buy=0.02,
        max_y_pred_to_sell=-0.02,
        transaction_fee=0.0,
        max_spending_for_a_day=3_000.0
    )

    # FullyConnectedNeuralNetwork
    # (money_made_lost_FullyConnectedNeuralNetwork,
    #  percent_made_lost_FullyConnectedNeuralNetwork,
    #  history_df_FullyConnectedNeuralNetwork, number_of_buys_FullyConnectedNeuralNetwork,
    #  number_of_sells_FullyConnectedNeuralNetwork) = run_model_simulation_backvalidation(
    #     df=data_experiment_train_and_validation_df,
    #     feature_cols_x=feature_cols_x,
    #     target_col="next_day_return",
    #     start_date_train_window=initial_train_start,
    #     end_date_train_window=initial_train_end,
    #     model=get_model(flag=5,
    #                     num_of_x_fields=len(feature_cols_x)),  # FullyConnectedNeuralNetwork
    #     price_col="close",
    #     initial_cash=10_000.0,
    #     min_y_pred_to_buy=0.02,
    #     max_y_pred_to_sell=-0.02,
    #     transaction_fee=0.0,
    #     max_spending_for_a_day=3_000.0
    # )

    money_made_comparison('XGBoost', money_made_lost_XGBoost,percent_made_lost_XGBoost,
                          number_of_buys_XGBoost, number_of_sells_XGBoost,
                          history_df_XGBoost,
                          'Linear regression',money_made_lost_Linear_Regression,
                          percent_made_lost_Linear_Regression,
                          number_of_buys_Linear_Regression, number_of_sells_Linear_Regression,
                          history_df_Linear_Regression)

    # money_made_comparison('rolling_avg_baseline', money_made_lost_rolling_avg_baseline,
    #                       percent_made_lost_rolling_avg_baseline,
    #                       number_of_buys_rolling_avg_baseline, number_of_sells_rolling_avg_baseline,
    #                       history_df_rolling_avg_baseline,
    #                       'previous_daily_return',money_made_lost_previous_daily_return,
    #                       percent_made_lost_previous_daily_return,
    #                       number_of_buys_previous_daily_return, number_of_sells_previous_daily_return,
    #                       history_df_previous_daily_return)

    # money_made_comparison('XGBoost', money_made_lost_XGBoost, percent_made_lost_XGBoost,
    #                       number_of_buys_XGBoost, number_of_sells_XGBoost,
    #                       history_df_XGBoost,
    #                       'FullyConnectedNeuralNetwork', money_made_lost_FullyConnectedNeuralNetwork,
    #                       percent_made_lost_FullyConnectedNeuralNetwork,
    #                       number_of_buys_FullyConnectedNeuralNetwork, number_of_sells_FullyConnectedNeuralNetwork,
    #                       history_df_FullyConnectedNeuralNetwork)




"""
# XGBoost
(money_made_lost_XGBoost,
 percent_made_lost_XGBoost,
 history_df_XGBoost, number_of_buys_XGBoost,
 number_of_sells_XGBoost) = run_model_simulation_backvalidation(
    df=data_experiment_train_and_validation_df,
    feature_cols_x=feature_cols_x,
    target_col="next_day_return",
    start_date_train_window=initial_train_start,
    end_date_train_window=initial_train_end,
    flag=1,  # XGBoost
    price_col="close",
    num_of_days=None,
    relevant_col_name="daily_return_percentage",
    initial_cash=10_000.0,
    min_y_pred_to_buy=0.02,
    max_y_pred_to_sell=-0.02,
    transaction_fee=0.0,
    max_spending_for_a_day=3_000.0
)


last run:
Money made/lost XGBoost: 600.1898299999993
Percent made/lost: XGBoost 6.0018982999999935
amount of stocks bought XGBoost: 56
amount of stocks sold XGBoost: 1
     validation_date next_trade_date        cash  portfolio_value_next_day  \
117 2026-03-06      2026-03-09  1353.62983               10262.05958   
118 2026-03-09      2026-03-10  1353.62983               10243.88981   
119 2026-03-10      2026-03-11  1353.62983               10629.79983   
120 2026-03-11      2026-03-12  1353.62983               10617.47983   
121 2026-03-12      2026-03-13  1353.62983               10600.18983   

     money_made_lost_step  percent_made_lost_total  
117              66.40959                 2.620596  
118             -18.16977                 2.438898  
119             385.91002                 6.297998  
120             -12.32000                 6.174798  
121             -17.29000                 6.001898  
number of trading days XGBoost: 122


rolling_avg on 5 days.
the rest is the same in all the models.
Money made/lost rolling_avg_baseline: 873.3105099999993
Percent made/lost: rolling_avg_baseline 8.733105099999994
amount of stocks bought rolling_avg_baseline: 614
amount of stocks sold rolling_avg_baseline: 460
     validation_date next_trade_date      cash  portfolio_value_next_day  \
117 2026-03-06      2026-03-09   2.89051               10948.11009   
118 2026-03-09      2026-03-10  54.55051               10722.67061   
119 2026-03-10      2026-03-11  26.89051               10810.79051   
120 2026-03-11      2026-03-12  26.89051               10763.02061   
121 2026-03-12      2026-03-13  26.89051               10873.31051   

     money_made_lost_step  percent_made_lost_total  
117             149.85967                 9.481101  
118            -225.43948                 7.226706  
119              88.11990                 8.107905  
120             -47.76990                 7.630206  
121             110.28990                 8.733105  
number of trading days rolling_avg_baseline: 122


Money made/lost previous_daily_return: -1292.655920000001
Percent made/lost: previous_daily_return -12.92655920000001
amount of stocks bought previous_daily_return: 2379
amount of stocks sold previous_daily_return: 2347
     validation_date next_trade_date      cash  portfolio_value_next_day  \
117 2026-03-06      2026-03-09  19.54394                8960.95412   
118 2026-03-09      2026-03-10  80.48394                8737.24394   
119 2026-03-10      2026-03-11   1.67406                8726.81388   
120 2026-03-11      2026-03-12  47.94406                8582.51402   
121 2026-03-12      2026-03-13   2.69406                8707.34408   

     money_made_lost_step  percent_made_lost_total  
117               3.44018               -10.390459  
118            -223.71018               -12.627561  
119             -10.43006               -12.731861  
120            -144.29986               -14.174860  
121             124.83006               -12.926559  
number of trading days previous_daily_return: 122
previous_daily_return made more money.


LOOKS LIKE rolling_avg_baseline IS THE BEST!!!!!!!!
"""