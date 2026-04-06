import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from data_preparing import unpickle_data
from data_preparing import pickle_file_path

# todo: see what is relevant and move to data prep,
#  I don't think any of it is relevant.
"""
This is the linear_regression model file
"""

"""
Because we found out that the price of a stock almost doesn't change, the R^2 is almost 1
the model predicts p(x) = p(x+1) all the time.
so we switch to log scale.
log(closed(x+1)/closed(x)) 
"""

# getting the data
"""
The names of the columns are: ['symbol', 'exchange', 'currency','date', 'open', 'high', 'low', 'close', 'volume'],
"""
five_thousand_days_data_df = unpickle_data(pickle_file_path)
five_thousand_days_data_df = five_thousand_days_data_df.rename(columns={'datetime': 'date'})

# one hot encoding of the symbol col
five_thousand_days_data_df['date'] = pd.to_datetime(five_thousand_days_data_df['date'])
min_date_all_symbols_have = five_thousand_days_data_df.groupby('symbol')['date'].min().max()

five_thousand_days_data_df = five_thousand_days_data_df.sort_values(['symbol', 'date'])
five_thousand_days_data_df['close_next_day'] = five_thousand_days_data_df.groupby('symbol')['close'].shift(-1)
five_thousand_days_data_df['log_return_next_day'] = np.log(
    five_thousand_days_data_df['close_next_day'] / five_thousand_days_data_df['close']
)
five_thousand_days_data_df = five_thousand_days_data_df.dropna(subset=['log_return_next_day'])
five_thousand_days_data_df = pd.get_dummies(five_thousand_days_data_df, columns=['symbol'], dtype=int)
print(five_thousand_days_data_df.columns)

# creating the linear regression model
linear_regression_model = LinearRegression()



# todo: check no data leaking, best k-fold way
"""
We want to create a K-fold like training section.
Because we can't just randomly split the data, because the time line is 
important, we'll create a moving window function for the training section.
total days - n
days to train on - x
train_test_split(x,n-x)
and we do that for many different x.
"""
# todo: give more weight to bigger x
# todo: for now, will train and test only on 1369 days ~ 3.75 years,
# todo: because it is the amount of rows of newest stock -> think if we should use all 5000 days


# company_row_counts = five_thousand_days_data_df.groupby('symbol').size().sort_values()
# print(company_row_counts)
# print(five_thousand_days_data_df.dtypes) -> checking the data types of the cols


def split_train_test(df, start_date, test_date):
    """
    Train on rows strictly before test_date.
    Test on rows whose date is test_date.
    """
    mask_train = (df['date'] >= start_date) & (df['date'] < test_date)
    train_df = df[mask_train]

    mask_test = (df['date'] == test_date)
    test_df = df[mask_test]

    return train_df, test_df


def split_x_y(train_df,test_df,x_fields=None,y_fields='daily_return_percentage'):
    """
        Split the data into training and testing sets.
        for now these are the fields.
        Left as function for future changes
        :param train_df: training set
        :param test_df: testing set
        :param x_fields: x fields to split
        :param y_fields: y fields to split
        :return: x_train, y_train, x_test, y_test
    """
    if x_fields is None:
        symbol_cols = [col for col in train_df.columns if col.startswith('symbol_')]
        x_fields = ['open', 'high', 'low', 'close', 'volume'] # + symbol_cols
    x_train = train_df[x_fields]
    y_train = train_df[y_fields]
    x_test = test_df[x_fields]
    y_test = test_df[y_fields]

    return x_train, y_train, x_test, y_test


def eval_model(df, train_start_date,train_end_date):
    """
    The function implements the expanding window idea
    and returns the avg R^2 of the model.
    :param df: the data frame
    :param start_date: the start date
    :param end_date: the end date
    :return: Test R² scores across expanding-window iterations
    """
    # each time train on x days
    # test the result on the x+1 day
    #  x = x + 1
    # and do it again until next_date == end_date
    all_dates = np.sort(
        df.loc[df['date'] >= train_start_date, 'date'].unique()
    )

    eval_dates = all_dates[1:]  # first test date is the second available date

    models_scores = []

    for current_date in eval_dates:
        current_date = pd.Timestamp(current_date)

        train_df, test_df = split_train_test(df, train_start_date, current_date)

        if train_df.empty or test_df.empty:
            continue

        x_train, y_train, x_test, y_test = split_x_y(train_df, test_df)

        linear_regression_model.fit(x_train, y_train)
        models_scores.append(linear_regression_model.score(x_test, y_test))

    # returning the avg of models
    if len(models_scores) == 0:
        return None
    return sum(models_scores) / len(models_scores)



######################## fitting the model#####################
print(eval_model(
    five_thousand_days_data_df,
    min_date_all_symbols_have,
    min_date_all_symbols_have
))

# def print_avg_daily_price_change(df):
#     """
#     Prints the average daily close-to-close change across stocks.
#     Uses percentage change and absolute percentage change.
#     """
#     temp_df = df.copy()
#
#     temp_df['date'] = pd.to_datetime(temp_df['date'])
#     temp_df = temp_df.sort_values(['symbol', 'date'])
#
#     temp_df['daily_pct_change'] = temp_df.groupby('symbol')['close'].pct_change()
#
#     avg_signed_change = temp_df['daily_pct_change'].mean()
#     avg_abs_change = temp_df['daily_pct_change'].abs().mean()
#
#     print("Average signed daily % change:", avg_signed_change)
#     print("Average absolute daily % change:", avg_abs_change)
#
#
# raw_df = unpickle_data(pickle_file_path)
# raw_df = raw_df.rename(columns={'datetime': 'date'})
#
# print_avg_daily_price_change(raw_df)


