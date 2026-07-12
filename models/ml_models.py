from __future__ import annotations
from typing import Optional

from typing import Any


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from models.model_base import Model

"""
Here we will write all the classes of the ML models.
Thy all inherit from the model_base class. 
"""


class XGBoostModel(Model):
    def __init__(self, objective: None = "reg:squarederror", n_estimators: int | None = 300,
                 max_depth: int | None = 3, learning_rate: float | None = 0.05,
                 subsample: float | None = 1.0, colsample_bytree: float | None = 0.8,
                 random_state: int | None =42 ,eval_metric: str | None = 'rmse') -> None:
                self.model = xgb.XGBRegressor(
                        objective=objective,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=random_state,
                        eval_metric=eval_metric
                    )

    def fit(self,x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on the training data.
        :param x_train:
        :param y_train:
        :return:
        """
        self.model.fit(x_train, y_train)

    def predict(
            self,
            x_test: pd.DataFrame,
            train_set: pd.DataFrame | None = None,
            val_set: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Predicts the predictions of the model on the x data.
        :param x_test:
        :return: The predictions.
        """
        y_pred = self.model.predict(x_test)
        return np.asarray(y_pred)






# todo: think about adding regulization to the model
class LinearRegressionModel(Model):
    def __init__(self):
        self.model = LinearRegression()


    def fit(self,x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on the training data.
        :param x_train:
        :param y_train:
        :return:
        """
        self.model.fit(x_train, y_train)

    def predict(
            self,
            x_test: pd.DataFrame,
            train_set: pd.DataFrame | None = None,
            val_set: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Predicts the predictions of the model on the x data.
        :param x_test:
        :param kwargs:
        :return:
        """
        return np.asarray(self.model.predict(x_test))