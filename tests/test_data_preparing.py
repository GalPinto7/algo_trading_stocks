"""
Code to test all the functions in the data_preparing module
"""
import pytest
import pandas as pd
import numpy as np
from data_preparing import * # todo: change later
import data_preparing as dp
from daily_stocks_prices_getter import calculate_values_of_rest_of_cols
from paths import DATA_DIR, TECH_UNIVERSE_CSV


"""
Tests we run in the cmd: python -m pytest -v test_data_preparing.py
"""

"""
PASSED ALL 47 TESTS!
"""

#################################### creating a pseudo_df to test the functions on ####################################
# 1. Define the parameters and companies picked from your list
companies = {
    'AAPL': ('NASDAQ', 'USD'),
    'MSFT': ('NASDAQ', 'USD'),
    'NVDA': ('NASDAQ', 'USD'),
    'IBM': ('NYSE', 'USD'),
    'PLTR': ('NYSE', 'USD')
}

# 2. Get the last 10 true market trading days (accounting for Memorial Day May 25, 2026)
trading_dates = [
    "2026-05-21", "2026-05-22",  # Thu, Fri
    # "2026-05-25" was Memorial Day (Market Closed)
    "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29",  # Tue to Fri
    "2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"  # Mon to Thu
]

# 3. Generate rows for every combination of symbol and date
rows = []
np.random.seed(42)  # For reproducible random stock numbers

for date in trading_dates:
    for symbol, (exchange, currency) in companies.items():
        # Generate realistic look-alike price data
        base_price = np.random.uniform(50, 400)
        open_p = round(base_price * np.random.uniform(0.98, 1.02), 2)
        close_p = round(base_price * np.random.uniform(0.98, 1.02), 2)
        high_p = round(max(open_p, close_p) * np.random.uniform(1.00, 1.03), 2)
        low_p = round(min(open_p, close_p) * np.random.uniform(0.97, 1.00), 2)
        volume = int(np.random.uniform(1_000_000, 50_000_000))
        next_day_return = round(np.random.uniform(-0.05, 0.05), 4)

        # Build raw record
        record = {
            'symbol': symbol,
            'exchange': exchange,
            'currency': currency,
            'date': date,
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': volume,
            'next_day_return': next_day_return,
            'exchange_NYSE': 1 if exchange == 'NYSE' else 0,
            'exchange_NASDAQ': 1 if exchange == 'NASDAQ' else 0,
        }

        # Explicitly initialize all your target symbol feature flags to 0
        all_target_symbols = [
            'AAPL', 'ACN', 'ADBE', 'ADP', 'AMAT', 'AMD', 'ANET', 'APH', 'AVGO', 'CDNS',
            'CRM', 'CRWD', 'CSCO', 'DDOG', 'DELL', 'DOCU', 'FTNT', 'HPE', 'HPQ', 'IBM',
            'INTC', 'INTU', 'KLAC', 'LRCX', 'MSFT', 'MU', 'NET', 'NOW', 'NVDA', 'ORCL',
            'PANW', 'PLTR', 'QCOM', 'ROP', 'SHOP', 'SNPS', 'TTD', 'TXN', 'UBER', 'ZM'
        ]

        for ts in all_target_symbols:
            record[f'symbol_{ts}'] = 1 if symbol == ts else 0

        rows.append(record)

# 4. Create DataFrame
pseudo_df = pd.DataFrame(rows)
########################################################################################################################



def test_project_paths_are_relative_to_project_data_folder():
    assert DATA_DIR.name == "data"
    assert TECH_UNIVERSE_CSV.name == "tech_universe.csv"
    assert TECH_UNIVERSE_CSV.parent == DATA_DIR


# 1. Define a pytest fixture to create a predictable mock DataFrame
@pytest.fixture
def mock_stock_df():
    """Generates a clean, non-random DataFrame mimicking our 10-day dataset."""
    data = [
        # AAPL Data
        {"symbol": "AAPL", "date": "2026-05-21", "close": 100.0},  # Thu
        {"symbol": "AAPL", "date": "2026-05-22", "close": 105.0},  # Fri
        # 2026-05-23 & 24 are Weekend (No records)
        # 2026-05-25 is Memorial Day Holiday (No record)
        {"symbol": "AAPL", "date": "2026-05-26", "close": 110.0},  # Tue
        {"symbol": "AAPL", "date": "2026-05-27", "close": 115.0},  # Wed
        {"symbol": "AAPL", "date": "2026-05-28", "close": 120.0},  # Thu

        # MSFT Data (for isolation check)
        {"symbol": "MSFT", "date": "2026-05-28", "close": 400.0}
    ]
    return pd.DataFrame(data)


# 2. Test Case: Normal expected behavior (7 calendar days back)
def test_percent_change_exact_match(mock_stock_df):
    # Current date: Thu, May 28 (Close: 120.0)
    # 7 calendar days ago: Thu, May 21 (Close: 100.0)
    # Expected change: ((120 - 100) / 100) * 100 = 20.0%

    current_date = pd.Timestamp("2026-05-28")
    result = get_percent_change_from_x_closing_days_ago(
        symbol="AAPL",
        date=current_date,
        start_days_ago=7,
        df=mock_stock_df
    )

    assert result == 20.0


# 3. Test Case: Looking back falls on a weekend/holiday, loop triggers
def test_percent_change_fallback_loop(mock_stock_df):
    # Current date: Tue, May 26 (Close: 110.0)
    # 4 calendar days ago: Fri, May 22 (Close: 105.0) -> Exact match!
    # Let's check a fallback: 3 calendar days ago was Sat, May 23 (No record).
    # The while loop should step back to Fri, May 22 automatically.
    # Expected calculation: ((110 - 105) / 105) * 100 = 4.7619%

    current_date = pd.Timestamp("2026-05-26")
    result = get_percent_change_from_x_closing_days_ago(
        symbol="AAPL",
        date=current_date,
        start_days_ago=3,  # Falls on Saturday, will increment to 4 (Friday)
        df=mock_stock_df
    )

    expected_return = ((110 - 105) / 105) * 100
    assert pytest.approx(result, rel=1e-4) == expected_return


# 4. Test Case: Exception Handling
def test_invalid_parameters(mock_stock_df):
    # Ensure it throws a ValueError if df is omitted
    with pytest.raises(ValueError, match="df must be provided."):
        get_percent_change_from_x_closing_days_ago("AAPL", pd.Timestamp("2026-05-28"))

    # Ensure it throws a ValueError if start_days_ago is 0 or negative
    with pytest.raises(ValueError, match="start_days_ago must be positive."):
        get_percent_change_from_x_closing_days_ago(
            "AAPL", pd.Timestamp("2026-05-28"), start_days_ago=0, df=mock_stock_df
        )




# def test_adding_col_with_values():




def test_add_days_ago_return_percentage_fast(mock_stock_df):
    # creating a fake df
    df = pd.DataFrame({
        "symbol": ["AAPL","AAPL","AAPL","MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
        ],
        "close": [100, 110, 121, 200, 220],
    })

    result = add_days_ago_return_percentage_fast(df = df, days_ago= 1, new_col_name = "ret_1")

    expected = [
        np.nan,  # AAPL first row has no previous AAPL day
        0.10,  # (110 - 100) / 100
        0.10,  # (121 - 110) / 110
        np.nan,  # MSFT first row has no previous MSFT day
        0.10,  # (220 - 200) / 200
    ]

    # after running the function it should be like these values,
    # so we check these really are the values we got.

    # does assert on all the values here at once
    np.testing.assert_allclose(
        # takes only the new column from the result DataFrame.
        actual = result["ret_1"].to_numpy(),
        # converts your expected list to a NumPy array:
        desired = np.array(expected),
        # Check that every value in actual is equal, or very close, to every value in expected
        equal_nan=True
    )


def test_keep_common_dates_only():
    df_fake = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL","AAPL","MSFT", "MSFT", "MSFT","MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-10",
            "2023-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-12",
        ],
        "close": [100, 110, 121, 200, 220,111,111,111],
    })

    df = keep_common_dates_only(df=df_fake)

    # After trimming, both should only have common dates:
    # '2024-01-01', '2024-01-02'

    for symbol in df['symbol'].unique():
        dates = df[df['symbol'] == symbol]['date'].tolist()
        assert dates == ["2024-01-01", "2024-01-02"]


def test_same_start_date_for_all_stocks():

    df_fake = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL",
                   "MSFT", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-10",
            "2023-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-12",
        ],
        "close": [100, 110, 121, 200, 220, 111, 112, 113],
    })

    df = same_start_date_for_all_stocks(df=df_fake)

    # ✅ Use Timestamp instead of string
    expected_start_date = pd.Timestamp("2024-01-01")

    start_dates = df.groupby('symbol')['date'].min()

    for symbol, start_date in start_dates.items():
        assert start_date == expected_start_date, (
            f"{symbol} start date {start_date} != {expected_start_date}"
        )

    # ✅ also compare as timestamps here
    assert (df['date'] >= expected_start_date).all()



def test_one_hot_encoding():
    df_fake = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL",
                   "MSFT", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-10",
            "2023-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-12",
        ],
        "close": [100, 110, 121, 200, 220, 111, 112, 113],
    })

    df = one_hot_encoding(df=df_fake, field_to_one_hot_encode="symbol")
    cols_list = df.columns.tolist()
    stocks_names_list = df["symbol"].unique().tolist()
    stocks_names_list = ["symbol_" + item for item in stocks_names_list]
    stocks_names_list.insert(0, "symbol")
    stocks_names_list.insert(1,"date")
    stocks_names_list.insert(2, "close")
    for col, stocks_names in zip(cols_list, stocks_names_list):
        assert col == stocks_names


def test_Populate_df1_in_df2_structure():
    df_fake_1 = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL",
                   "MSFT", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-10",
            "2023-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-12",
        ],
        "close": [100, 110, 121, 200, 220, 111, 112, 113],
    })

    df_fake_2 = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL",
                   "MSFT", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-10",
            "2023-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-12",
        ],
        "open": [100, 110, 121, 200, 220, 111, 112, 113],
    })

    # check 1
    df_fake_3 = pd.DataFrame({})
    df_fake_1_cols = df_fake_1.columns.tolist()
    df_fake_2 = Populate_df1_in_df2_structure(df_fake_2, df_fake_1)


    with pytest.raises(ValueError, match="df1 can't be empty!"):
        Populate_df1_in_df2_structure(df1=df_fake_3, df2=df_fake_1)

    df_fake_1_cols = df_fake_1.columns.tolist()
    df_fake_2_cols = df_fake_2.columns.tolist()
    df_fake_3_cols = df_fake_3.columns.tolist()
    for col_1,col_2,col_3 in zip(df_fake_1_cols, df_fake_2_cols,df_fake_3_cols):
        assert col_2 == col_1
        assert col_3 == col_1


def test_add_row_of_new_df_to_og_df(): # TESTED
    df1 = pd.DataFrame({"names": ["gal","shay","eli"],
           "age": [26,28,60]})

    df2 = pd.DataFrame({"names": ["shifo","andrea"],
           "age": [10,60]})

    df3 = pd.DataFrame({"name":["gal"]})

    df_test_1 = add_row_of_new_df_to_og_df(new_df=df2, og_df=df1)
    df_test_2 = df_test_1
    df_test_2_cols = df_test_2.columns.tolist()

    expected_1_names = pd.Series(["gal","shay","eli","shifo","andrea"], name="names")
    expected_1_ages = pd.Series([26,28,60,10,60], name="age")

    # check 1:
    # does assert on all the values here at once
    pd.testing.assert_series_equal(expected_1_names,df_test_1["names"])

    pd.testing.assert_series_equal(expected_1_ages, df_test_1["age"])

    # check 2: doesn't mess up the order of the cols
    expected_cols = ["names","age"]
    assert df_test_2_cols == expected_cols

    # --- Check 3: Error handling for bad structure (df1 vs df3) ---
    # This wrapper checks if the exact ValueError with your custom message is raised
    with pytest.raises(ValueError, match="The dataframes must have the exact same columns in the same order."):
        add_row_of_new_df_to_og_df(new_df=df3, og_df=df1)


def test_remove_duplicates_based_on_fields():
    df = pd.DataFrame({
        "name":["gal","shay","gal","shay"],
        "age":[1,1,1,2]
    })

    df = remove_duplicates_based_on_fields(df,["name","age"])
    wanted_names = ["gal","shay","shay"]
    wanted_ages = [1,1,2]
    # pd.testing.assert_series_equal(expected_1_names,df_test_1["names"])
    assert df["name"].tolist() == wanted_names
    assert df["age"].tolist() == wanted_ages


#######################################################more tests#######################################################
def test_pickling_func_and_unpickle_data(tmp_path):
    data = {"symbols": ["AAPL", "MSFT"], "values": [1, 2]}
    file_path = tmp_path / "test_data.pkl"

    pickling_func(data=data, file_path=str(file_path))
    loaded_data = unpickle_data(str(file_path))

    assert loaded_data == data


def test_pickling_func_invalid_inputs(tmp_path):
    file_path = tmp_path / "test_data.pkl"

    with pytest.raises(ValueError, match="data can't be None"):
        pickling_func(data=None, file_path=str(file_path))

    with pytest.raises(ValueError, match="file_path can't be None"):
        pickling_func(data={"a": 1}, file_path=None)

    with pytest.raises(ValueError, match="file_path can't be None"):
        unpickle_data(file_path=None)


def test_loader_functions_reject_bad_where_the_code_runs():
    bad_value = 3

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_top_40_tech_companies_csv(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_top_40_tech_companies_names(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_Earliest_Timestamps_top_40_tech_companies_daily(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_Earliest_Timestamps_top_40_tech_companies_hourly(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_five_thousand_days_data_df(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_five_thousand_days_data_experiment_df(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_data_experiment_train_and_validation_df(bad_value)

    with pytest.raises(ValueError, match="Where code must be either 1 or 2"):
        load_data_experiment_test_df(bad_value)


def test_load_top_40_tech_companies_names_uses_unpickle_data(monkeypatch):
    def fake_unpickle_data(file_path):
        return ["AAPL", "MSFT"]

    monkeypatch.setattr(dp, "unpickle_data", fake_unpickle_data)

    result = dp.load_top_40_tech_companies_names(where_the_code_runs=1)

    assert result == ["AAPL", "MSFT"]


def test_adding_col_with_values():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": ["2024-01-01", "2024-01-01"],
    })

    def fake_function(symbol, date):
        if symbol == "AAPL":
            return 10
        return 20

    result = adding_col_with_values(
        df=df,
        new_col_name="new_value",
        function=fake_function
    )

    assert result["new_value"].tolist() == [10, 20]


def test_adding_col_with_values_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "existing_col": [1],
    })

    with pytest.raises(ValueError, match="new_col_name can not be null"):
        adding_col_with_values(df=df, new_col_name=None, function=lambda symbol, date: 1)

    with pytest.raises(ValueError, match="The df has a col with this name already"):
        adding_col_with_values(df=df, new_col_name="existing_col", function=lambda symbol, date: 1)


def test_add_days_ago_return_percentage_fast_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "close": [100],
    })

    with pytest.raises(ValueError, match="df must be provided"):
        add_days_ago_return_percentage_fast(df=None, days_ago=1, new_col_name="ret_1")

    with pytest.raises(ValueError, match="days_ago must be positive"):
        add_days_ago_return_percentage_fast(df=df, days_ago=0, new_col_name="ret_1")

    with pytest.raises(ValueError, match="new_col_name can not be null"):
        add_days_ago_return_percentage_fast(df=df, days_ago=1, new_col_name=None)


def test_fast_perf_x_trading_days_ago():
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        "close": [100, 110, 120, 150, 200, 220, 300],
    })

    result = fast_perf_x_trading_days_ago(
        df=df,
        days_ago=2,
        col_name="ret_2"
    )

    expected = [
        np.nan,          # AAPL day 1
        np.nan,          # AAPL day 2
        0.20,            # (120 - 100) / 100
        150 / 110 - 1,   # (150 - 110) / 110
        np.nan,          # MSFT day 1
        np.nan,          # MSFT day 2
        0.50,            # (300 - 200) / 200
    ]

    np.testing.assert_allclose(
        result["ret_2"].to_numpy(),
        np.array(expected),
        equal_nan=True
    )


def test_fast_perf_x_trading_days_ago_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "close": [100],
    })

    with pytest.raises(ValueError, match="Df can't be none"):
        fast_perf_x_trading_days_ago(df=None, days_ago=1, col_name="ret_1")

    with pytest.raises(ValueError, match="days ago can't be none"):
        fast_perf_x_trading_days_ago(df=df, days_ago=0, col_name="ret_1")

    with pytest.raises(ValueError, match="col_name can not be null"):
        fast_perf_x_trading_days_ago(df=df, days_ago=1, col_name=None)


def test_fast_perf_x_trading_days_ago_for_last_date():
    historic_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        "close": [100, 110, 120, 200, 220, 240],
    })

    new_day_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": ["2024-01-04", "2024-01-04"],
        "close": [150, 300],
    })

    result = fast_perf_x_trading_days_ago_for_last_date(
        new_day_df=new_day_df,
        historic_df=historic_df,
        days_ago=2,
        name_of_col="ret_2",
    )

    assert len(result) == 8

    aapl_new_row = result[
        (result["symbol"] == "AAPL") &
        (result["date"] == pd.Timestamp("2024-01-04"))
    ].iloc[0]

    msft_new_row = result[
        (result["symbol"] == "MSFT") &
        (result["date"] == pd.Timestamp("2024-01-04"))
    ].iloc[0]

    assert np.isclose(aapl_new_row["ret_2"], (150 - 110) / 110)
    assert np.isclose(msft_new_row["ret_2"], (300 - 220) / 220)

    # Historic rows should stay in the output.
    assert pd.Timestamp("2024-01-01") in result["date"].tolist()


def test_fast_perf_x_trading_days_ago_for_last_date_drops_duplicates():
    historic_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": ["2024-01-01", "2024-01-02"],
        "close": [100, 110],
    })

    # Same symbol/date as an old row. The new row should replace the old duplicate.
    new_day_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-02"],
        "close": [120],
    })

    result = fast_perf_x_trading_days_ago_for_last_date(
        new_day_df=new_day_df,
        historic_df=historic_df,
        days_ago=1,
        name_of_col="ret_1",
    )

    duplicated_rows = result[
        (result["symbol"] == "AAPL") &
        (result["date"] == pd.Timestamp("2024-01-02"))
    ]

    assert len(duplicated_rows) == 1
    assert duplicated_rows["close"].iloc[0] == 120


def test_fast_perf_x_trading_days_ago_for_last_date_invalid_inputs():
    historic_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "close": [100],
    })

    new_day_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-02"],
        "close": [110],
    })

    with pytest.raises(ValueError, match="new_day_df is empty"):
        fast_perf_x_trading_days_ago_for_last_date(
            new_day_df=pd.DataFrame(),
            historic_df=historic_df,
            days_ago=1,
            name_of_col="ret_1",
        )

    with pytest.raises(ValueError, match="historic_df is empty"):
        fast_perf_x_trading_days_ago_for_last_date(
            new_day_df=new_day_df,
            historic_df=pd.DataFrame(),
            days_ago=1,
            name_of_col="ret_1",
        )

    with pytest.raises(ValueError, match="at least one day"):
        fast_perf_x_trading_days_ago_for_last_date(
            new_day_df=new_day_df,
            historic_df=historic_df,
            days_ago=0,
            name_of_col="ret_1",
        )

    bad_new_day_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-02"],
    })

    with pytest.raises(ValueError, match="close must be in new_day_df"):
        fast_perf_x_trading_days_ago_for_last_date(
            new_day_df=bad_new_day_df,
            historic_df=historic_df,
            days_ago=1,
            name_of_col="ret_1",
        )


def test_relative_field_x_days():
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        "volume": [100, 200, 300, 400, 10, 20, 30],
    })

    result = relative_field_x_days(
        df=df,
        num_of_days=2,
        new_col_name="relative_volume_2_days",
        relevant_col_name="volume",
    )

    expected = [
        np.nan,
        np.nan,
        300 / 150,
        400 / 250,
        np.nan,
        np.nan,
        30 / 15,
    ]

    np.testing.assert_allclose(
        result["relative_volume_2_days"].to_numpy(),
        np.array(expected),
        equal_nan=True
    )


def test_relative_field_x_days_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "volume": [100],
    })

    with pytest.raises(ValueError, match="Df can't be None"):
        relative_field_x_days(None, 2, "relative_volume", "volume")

    with pytest.raises(ValueError, match="num_of_days must be positive"):
        relative_field_x_days(df, 0, "relative_volume", "volume")

    with pytest.raises(ValueError, match="new_col_name can't be None"):
        relative_field_x_days(df, 2, None, "volume")

    with pytest.raises(ValueError, match="relevant_col_name can't be None"):
        relative_field_x_days(df, 2, "relative_volume", None)


def test_calculate_rolling_mean_for_x_days():
    df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
        ],
        "close": [100, 110, 120, 200, 220],
    })

    result = calculate_rolling_mean_for_x_days(
        df=df,
        field_name="close",
        num_of_days=2,
        new_col_name="SMA_2",
        after_col_put_new_col="close",
    )

    expected = [np.nan, 105, 115, np.nan, 210]

    np.testing.assert_allclose(
        result["SMA_2"].to_numpy(),
        np.array(expected),
        equal_nan=True
    )

    columns = result.columns.tolist()
    assert columns[columns.index("close") + 1] == "SMA_2"


def test_calculate_rolling_mean_for_x_days_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-01"],
        "close": [100],
    })

    with pytest.raises(ValueError, match="Df can't be None"):
        calculate_rolling_mean_for_x_days(None, "close", 2, "SMA_2")

    with pytest.raises(ValueError, match="Field name can't be None"):
        calculate_rolling_mean_for_x_days(df, None, 2, "SMA_2")

    with pytest.raises(ValueError, match="num_of_days must be positive"):
        calculate_rolling_mean_for_x_days(df, "close", 0, "SMA_2")

    with pytest.raises(ValueError, match="new_col_name can't be None"):
        calculate_rolling_mean_for_x_days(df, "close", 2, None)


def test_add_sma_x_gap_percent():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "close": [110, 180],
        "SMA_20": [100, 200],
    })

    result = add_sma_x_gap_percent(
        df=df,
        sma_col_name="SMA_20",
        new_col_name="SMA_20_gap_percent",
        after_col_put_new_col="SMA_20",
    )

    expected = [10.0, -10.0]

    np.testing.assert_allclose(
        result["SMA_20_gap_percent"].to_numpy(),
        np.array(expected)
    )

    columns = result.columns.tolist()
    assert columns[columns.index("SMA_20") + 1] == "SMA_20_gap_percent"


def test_add_sma_x_gap_percent_invalid_inputs():
    df = pd.DataFrame({
        "close": [110],
        "SMA_20": [100],
    })

    with pytest.raises(ValueError, match="Df can't be None"):
        add_sma_x_gap_percent(None, "SMA_20", "gap")

    with pytest.raises(ValueError, match="sma_col_name can't be None"):
        add_sma_x_gap_percent(df, None, "gap")

    with pytest.raises(ValueError, match="new_col_name can't be None"):
        add_sma_x_gap_percent(df, "SMA_20", None)


def test_move_col_position_in_df_by_index():
    df = pd.DataFrame({
        "a": [1],
        "b": [2],
        "c": [3],
    })

    result = move_col_position_in_df(
        df=df.copy(),
        name_of_col_to_move="c",
        index_of_new_position=0,
    )

    assert result.columns.tolist() == ["c", "a", "b"]


def test_move_col_position_in_df_after_col():
    df = pd.DataFrame({
        "a": [1],
        "b": [2],
        "c": [3],
    })

    result = move_col_position_in_df(
        df=df.copy(),
        name_of_col_to_move="a",
        name_of_col_to_move_after="b",
    )

    assert result.columns.tolist() == ["b", "a", "c"]


def test_move_col_position_in_df_invalid_inputs():
    df = pd.DataFrame({
        "a": [1],
        "b": [2],
    })

    with pytest.raises(ValueError, match="exactly one"):
        move_col_position_in_df(
            df=df.copy(),
            name_of_col_to_move="a",
            index_of_new_position=0,
            name_of_col_to_move_after="b",
        )

    with pytest.raises(ValueError, match="exactly one"):
        move_col_position_in_df(
            df=df.copy(),
            name_of_col_to_move="a",
        )


def test_add_col_intraday_range():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": ["2024-01-01", "2024-01-01"],
        "high": [120, 220],
        "low": [100, 200],
        "close": [110, 210],
    })

    result = add_col_intraday_range(df)

    expected = [
        ((120 - 100) / 110) * 100,
        ((220 - 200) / 210) * 100,
    ]

    np.testing.assert_allclose(
        result["intraday_range_percent"].to_numpy(),
        np.array(expected)
    )


def test_add_col_intraday_range_invalid_input():
    with pytest.raises(ValueError, match="df can't be None"):
        add_col_intraday_range(None)


def test_create_df_of_x_percent_of_the_rows_start_and_end():
    df = pd.DataFrame({
        "value": list(range(10))
    })

    start_result = create_df_of_x_percent_of_the_rows(
        df=df,
        percent_to_take=30,
        from_end_or_start="start"
    )

    end_result = create_df_of_x_percent_of_the_rows(
        df=df,
        percent_to_take=30,
        from_end_or_start="end"
    )

    assert start_result["value"].tolist() == [0, 1, 2]
    assert end_result["value"].tolist() == [7, 8, 9]


def test_create_df_of_x_percent_of_the_rows_rounds_up():
    df = pd.DataFrame({
        "value": list(range(10))
    })

    result = create_df_of_x_percent_of_the_rows(
        df=df,
        percent_to_take=5,
        from_end_or_start="start"
    )

    assert len(result) == 1
    assert result["value"].tolist() == [0]


def test_create_df_of_x_percent_of_the_rows_invalid_inputs():
    df = pd.DataFrame({
        "value": list(range(10))
    })

    with pytest.raises(ValueError, match="df can't be None"):
        create_df_of_x_percent_of_the_rows(None, 50)

    with pytest.raises(ValueError, match="percent must be between 0 to 100"):
        create_df_of_x_percent_of_the_rows(df, 101)

    with pytest.raises(ValueError, match="from_end_or_start_flag must be start or end"):
        create_df_of_x_percent_of_the_rows(df, 50, "middle")


def test_split_df_to_train_val_test_with_explicit_test_percent():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "value": list(range(10)),
    })

    train_set, validation_set, test_set = split_df_to_train_val_test(
        df=df,
        percent_for_train=50,
        percent_for_val=30,
        percent_for_test=20,
    )

    assert len(train_set) == 5
    assert len(validation_set) == 3
    assert len(test_set) == 2

    assert train_set["value"].tolist() == [0, 1, 2, 3, 4]
    assert validation_set["value"].tolist() == [5, 6, 7]
    assert test_set["value"].tolist() == [8, 9]

    assert train_set.index.intersection(validation_set.index).empty
    assert train_set.index.intersection(test_set.index).empty
    assert validation_set.index.intersection(test_set.index).empty


def test_split_df_to_train_val_test_when_test_percent_is_none():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "value": list(range(10)),
    })

    train_set, validation_set, test_set = split_df_to_train_val_test(
        df=df,
        percent_for_train=60,
        percent_for_val=20,
        percent_for_test=None,
    )

    assert len(train_set) == 6
    assert len(validation_set) == 2
    assert len(test_set) == 2


def test_split_df_to_train_val_test_invalid_inputs():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "value": list(range(10)),
    })

    with pytest.raises(ValueError, match="df can't be None"):
        split_df_to_train_val_test(None, 50, 30, 20)

    with pytest.raises(ValueError, match="sum to 100"):
        split_df_to_train_val_test(df, 50, 30, 30)

    with pytest.raises(ValueError, match="sum to 100"):
        split_df_to_train_val_test(df, 90, 20, None)


def test_data_split_to_train_and_validation():
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        "date": [
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ],
        "close": [100, 200, 110, 220, 120, 240],
    })

    train_set, validation_set = data_split_to_train_and_validation(
        df=df,
        start_date_train_window=pd.Timestamp("2024-01-01"),
        validation_date=pd.Timestamp("2024-01-03"),
    )

    assert train_set["date"].dt.normalize().unique().tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]

    assert validation_set["date"].dt.normalize().unique().tolist() == [
        pd.Timestamp("2024-01-03")
    ]

    assert len(train_set) == 4
    assert len(validation_set) == 2


def test_data_split_to_train_and_validation_invalid_inputs():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "date": ["2024-01-02"],
        "close": [100],
    })

    with pytest.raises(ValueError, match="df can't be None"):
        data_split_to_train_and_validation(
            df=None,
            start_date_train_window=pd.Timestamp("2024-01-01"),
            validation_date=pd.Timestamp("2024-01-02"),
        )

    with pytest.raises(ValueError, match="start_date_train_window"):
        data_split_to_train_and_validation(
            df=df,
            start_date_train_window=pd.Timestamp("2024-01-01"),
            validation_date=pd.Timestamp("2024-01-02"),
        )

import numpy as np
import pandas as pd

from daily_stocks_prices_getter import calculate_values_of_rest_of_cols


def create_old_df_for_daily_update_test() -> pd.DataFrame:
    rows = []

    symbols = {
        "AAPL": ("NASDAQ", "USD", 100),
        "MSFT": ("NASDAQ", "USD", 200),
    }

    # 30 old trading days.
    dates = pd.bdate_range("2026-01-01", periods=30)

    for symbol, (exchange, currency, base_close) in symbols.items():
        for i, date in enumerate(dates):
            close = base_close + i
            rows.append({
                "symbol": symbol,
                "exchange": exchange,
                "currency": currency,
                "date": date,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + i,

                # Existing feature cols.
                # We put fake old values to check they are not destroyed.
                "ret_1": -999.0,
                "ret_2": -999.0,
                "ret_3": -999.0,
                "ret_5": -999.0,
                "ret_7": -999.0,
                "ret_10": -999.0,
                "ret_30": -999.0,
                "relative_volume_20_days": -999.0,
            })

    return pd.DataFrame(rows)


def create_new_df_for_daily_update_test() -> pd.DataFrame:
    rows = []

    new_date = pd.Timestamp("2026-02-12")

    new_values = {
        "AAPL": ("NASDAQ", "USD", 130),
        "MSFT": ("NASDAQ", "USD", 230),
    }

    for symbol, (exchange, currency, close) in new_values.items():
        rows.append({
            "symbol": symbol,
            "exchange": exchange,
            "currency": currency,
            "date": new_date,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1030,
        })

    return pd.DataFrame(rows)


def test_calculate_values_of_rest_of_cols_keeps_old_rows_and_adds_new_rows():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df,
    )

    # Old rows + new rows.
    assert len(result) == len(old_df) + len(new_df)

    # No duplicate symbol/date rows.
    duplicated_count = result.duplicated(subset=["symbol", "date"]).sum()
    assert duplicated_count == 0

    # Every new row exists in the final result.
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    new_date = pd.Timestamp("2026-02-12").normalize()

    new_rows_in_result = result[result["date"] == new_date]

    assert len(new_rows_in_result) == len(new_df)
    assert set(new_rows_in_result["symbol"]) == {"AAPL", "MSFT"}


def test_calculate_values_of_rest_of_cols_does_not_destroy_old_feature_values():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df
    )

    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    old_dates = pd.to_datetime(old_df["date"]).dt.normalize().unique()

    old_rows_in_result = result[result["date"].isin(old_dates)]

    # Old fake values should remain unchanged.
    # This checks that the function did not recalculate/overwrite old rows.
    cols_to_check = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_7",
        "ret_10",
        "ret_30",
        "relative_volume_20_days",
    ]

    for col in cols_to_check:
        assert (old_rows_in_result[col] == -999.0).all()


def test_calculate_values_of_rest_of_cols_calculates_new_row_values():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df,
    )

    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    new_date = pd.Timestamp("2026-02-12").normalize()

    aapl_new_row = result[
        (result["symbol"] == "AAPL") &
        (result["date"] == new_date)
    ].iloc[0]

    # AAPL old closes are 100, 101, ..., 129.
    # New close is 130.
    assert np.isclose(aapl_new_row["ret_1"], (130 - 129) / 129)
    assert np.isclose(aapl_new_row["ret_2"], (130 - 128) / 128)
    assert np.isclose(aapl_new_row["ret_3"], (130 - 127) / 127)
    assert np.isclose(aapl_new_row["ret_5"], (130 - 125) / 125)
    assert np.isclose(aapl_new_row["ret_7"], (130 - 123) / 123)
    assert np.isclose(aapl_new_row["ret_10"], (130 - 120) / 120)

    # There are only 30 previous rows.
    # For ret_30, shift(30) means the new row compares to the first old row.
    assert np.isclose(aapl_new_row["ret_30"], (130 - 100) / 100)

    # relative_volume_20_days = current volume / average previous 20 volumes
    # Previous 20 volumes are 1010..1029
    expected_avg_volume = np.mean(list(range(1010, 1030)))
    expected_relative_volume = 1030 / expected_avg_volume




def create_old_df_for_daily_update_test() -> pd.DataFrame:
    rows = []

    symbols = {
        "AAPL": ("NASDAQ", "USD", 100),
        "MSFT": ("NASDAQ", "USD", 200),
    }

    # 30 old trading days.
    dates = pd.bdate_range("2026-01-01", periods=30)

    for symbol, (exchange, currency, base_close) in symbols.items():
        for i, date in enumerate(dates):
            close = base_close + i
            rows.append({
                "symbol": symbol,
                "exchange": exchange,
                "currency": currency,
                "date": date,
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000 + i,

                # Existing feature cols.
                # We put fake old values to check they are not destroyed.
                "ret_1": -999.0,
                "ret_2": -999.0,
                "ret_3": -999.0,
                "ret_5": -999.0,
                "ret_7": -999.0,
                "ret_10": -999.0,
                "ret_30": -999.0,
                "relative_volume_20_days": -999.0,
            })

    return pd.DataFrame(rows)


def create_new_df_for_daily_update_test() -> pd.DataFrame:
    rows = []

    new_date = pd.Timestamp("2026-02-12")

    new_values = {
        "AAPL": ("NASDAQ", "USD", 130),
        "MSFT": ("NASDAQ", "USD", 230),
    }

    for symbol, (exchange, currency, close) in new_values.items():
        rows.append({
            "symbol": symbol,
            "exchange": exchange,
            "currency": currency,
            "date": new_date,
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1030,
        })

    return pd.DataFrame(rows)


def test_calculate_values_of_rest_of_cols_keeps_old_rows_and_adds_new_rows():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df,
    )

    # Old rows + new rows.
    assert len(result) == len(old_df) + len(new_df)

    # No duplicate symbol/date rows.
    duplicated_count = result.duplicated(subset=["symbol", "date"]).sum()
    assert duplicated_count == 0

    # Every new row exists in the final result.
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    new_date = pd.Timestamp("2026-02-12").normalize()

    new_rows_in_result = result[result["date"] == new_date]

    assert len(new_rows_in_result) == len(new_df)
    assert set(new_rows_in_result["symbol"]) == {"AAPL", "MSFT"}


def test_calculate_values_of_rest_of_cols_does_not_destroy_old_feature_values():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df,
    )

    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    old_dates = pd.to_datetime(old_df["date"]).dt.normalize().unique()

    old_rows_in_result = result[result["date"].isin(old_dates)]

    # Old fake values should remain unchanged.
    # This checks that the function did not recalculate/overwrite old rows.
    cols_to_check = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_7",
        "ret_10",
        "ret_30",
    ]

    for col in cols_to_check:
        assert (old_rows_in_result[col] == -999.0).all()


    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    new_date = pd.Timestamp("2026-02-12").normalize()

    aapl_new_row = result[
        (result["symbol"] == "AAPL") &
        (result["date"] == new_date)
    ].iloc[0]

    # AAPL old closes are 100, 101, ..., 129.
    # New close is 130.
    assert np.isclose(aapl_new_row["ret_1"], (130 - 129) / 129)
    assert np.isclose(aapl_new_row["ret_2"], (130 - 128) / 128)
    assert np.isclose(aapl_new_row["ret_3"], (130 - 127) / 127)
    assert np.isclose(aapl_new_row["ret_5"], (130 - 125) / 125)
    assert np.isclose(aapl_new_row["ret_7"], (130 - 123) / 123)
    assert np.isclose(aapl_new_row["ret_10"], (130 - 120) / 120)

    # There are only 30 previous rows.
    # For ret_30, shift(30) means the new row compares to the first old row.
    assert np.isclose(aapl_new_row["ret_30"], (130 - 100) / 100)



def test_calculate_values_of_rest_of_cols_existing_col_names_are_not_a_problem():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    # Make new_df already have these columns.
    # This simulates your fear: "new_col_name already exists in the df".
    for col in [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_7",
        "ret_10",
        "ret_30",
    ]:
        new_df[col] = np.nan

    result = calculate_values_of_rest_of_cols(
        old_df=old_df,
        new_df=new_df,
    )

    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    new_date = pd.Timestamp("2026-02-12").normalize()

    new_rows_in_result = result[result["date"] == new_date]

    # Existing column names should not create duplicate columns.
    assert result.columns.tolist().count("ret_1") == 1
    assert result.columns.tolist().count("ret_10") == 1
    assert result.columns.tolist().count("ret_30") == 1

    # New rows should have calculated values, not remain NaN.
    assert new_rows_in_result["ret_1"].notna().all()
    assert new_rows_in_result["ret_10"].notna().all()
    assert new_rows_in_result["ret_30"].notna().all()


def test_calculate_values_of_rest_of_cols_empty_inputs_raise_error():
    old_df = create_old_df_for_daily_update_test()
    new_df = create_new_df_for_daily_update_test()

    empty_df = pd.DataFrame()

    try:
        calculate_values_of_rest_of_cols(
            old_df=empty_df,
            new_df=new_df,
        )
        assert False, "Expected ValueError for empty old_df"
    except ValueError:
        pass

    try:
        calculate_values_of_rest_of_cols(
            old_df=old_df,
            new_df=empty_df,
        )
        assert False, "Expected ValueError for empty new_df"
    except ValueError:
        pass

