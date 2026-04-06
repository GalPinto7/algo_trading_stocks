from __future__ import annotations
from typing import Optional

from typing import Any

import pandas as pd
import numpy as np
from models.model_base import Model


class DumbModel(Model):
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """ no fit in this model"""
        pass

    def predict(self, x_test: pd.DataFrame, train_set: pd.DataFrame | None = None,
                val_set: pd.DataFrame | None = None) -> np.ndarray:
        """ we just predict 0s """
        # Using shape[0] is safer than len(y_val)
        return np.zeros(x_test.shape[0])



class PreviousDayReturnModel(Model):

    def __init__(self, relevant_col_name: str = "daily_return_percentage") -> None:
        self.relevant_col_name = relevant_col_name

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Baselines do not require a training fit."""
        pass

    def predict(
            self,
            x_test: pd.DataFrame,
            train_set: pd.DataFrame | None = None,
            val_set: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Predict each stock's next-day return as its previous daily return
        from the last date in the train set.
        """
        if train_set is None or val_set is None:
            raise ValueError("PreviousReturnModel requires train_set and validation_set.")

        if self.relevant_col_name not in train_set.columns:
            raise ValueError(f'{self.relevant_col_name} is not a column in the df!')

        # copy to not change the df itself
        train_set = train_set.copy()
        validation_set = val_set.copy()

        # make sure the data format is right
        train_set["date"] = pd.to_datetime(train_set["date"])
        validation_set["date"] = pd.to_datetime(validation_set["date"])

        # max date of the train set is the date we take the daily return from
        last_train_date = train_set["date"].max()

        # taking the daily return
        last_day_returns = (
            train_set[train_set["date"] == last_train_date]
            .set_index("symbol")[self.relevant_col_name]
        )

        y_pred = validation_set["symbol"].map(last_day_returns).fillna(0).to_numpy()

        return y_pred




class RollingAvgModel(Model):
    def __init__(self, num_of_days: int, relevant_col_name: str = "daily_return_percentage") -> None:
        """
        The builder.
        :param num_of_days: The number of days we use to calculate the AVG.
        :param relevant_col_name: The field we calculate the AVG on.
        """
        self.num_of_days = num_of_days
        self.relevant_col_name = relevant_col_name

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """ There is no fit phase for this model """
        pass

    def predict(
    self,
    x_test: pd.DataFrame,
    train_set: pd.DataFrame | None = None,
    val_set: pd.DataFrame | None = None
) -> np.ndarray:
        """
        Return a np.ndarray of avg daily_return.
        number of days, and relevant_col_name are fields in the model, no need to write in the function.
        :param train_set: The training set
        :param val_set: The validation set
        :return: np.ndarray of avg daily_return
        """

        if train_set is None or val_set is None:
            raise ValueError("RollingAvgModel requires train_set and validation_set.")

        if self.relevant_col_name not in train_set.columns:
            raise ValueError(f'{self.relevant_col_name} is not a column in the df!')

        train_set = train_set.copy()
        validation_set = val_set.copy()

        train_set['date'] = pd.to_datetime(train_set['date'])
        validation_set['date'] = pd.to_datetime(validation_set['date'])

        train_set = train_set.sort_values(["symbol", "date"]).reset_index(drop=True)

        # rolling average per symbol, using the last num_of_days known values
        train_set["rolling_avg_pred"] = (
            train_set.groupby("symbol")[self.relevant_col_name]
            .transform(lambda s: s.rolling(window=self.num_of_days, min_periods=self.num_of_days).mean())
        )

        last_train_date = train_set["date"].max()

        last_rolling_avg = (
            train_set[train_set["date"] == last_train_date]
            .set_index("symbol")["rolling_avg_pred"]
        )

        y_pred = validation_set["symbol"].map(last_rolling_avg).to_numpy()

        return y_pred



