from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional
from typing import Any

"""
This is an abstract base class and should not be used directly.
It is the definition of a model and the function it has.

A model needs 3 functions:
1. predict (predict function)
2. train_model (fit)

No '__init__' because it is an abstract class, can't create a obj. 
"""

class Model(ABC):
    @abstractmethod # convention
    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the model on the x_train, y_train.
        :param x_train:
        :param y_train:
        :return:
        """
        pass

    @abstractmethod # convention
    def predict(self, x_test: pd.DataFrame, train_set: pd.DataFrame | None = None,
                val_set: pd.DataFrame | None = None) -> np.ndarray:
        """
        Predicts the values of x_test using the trained model.
        we have the train_set and val_set for the models like the rolling_avg and
        previous_day that need them.
        :param x_test:
        :param train_set:
        :param val_set:
        :return:
        """
        pass





