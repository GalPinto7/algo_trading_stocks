from __future__ import annotations
from typing import Optional

from typing import Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from models.model_base import Model


"""
We will write here the NN models classes.
1. 
"""

# todo: maybe need more stocks -> get hourly records


# todo: features scaling !!!!!!
#  is it relevant for the all the NN models? ML models?
class FullyConnectedNeuralNetwork(Model):
    # todo: think about regularize
    def __init__(self, num_of_fields: int, number_of_layers: int | None = 3,
                 number_of_neurons_in_the_first_layer: int | None = 64,activation_function: str = 'relu',
                 probability_flag: int | None = 0, dived_num_of_neurons_by_each_layer: int | None = 2,
                 epochs: int | None = 10, batch_size: int | None = 32, verbose: int | None = 0):
        """
        Creates the FullyConnectedNeuralNetwork object.
        We use the abstract class 'Model' we defined, so we have to write the same functions as the functions
        def there.
        There are 2, fit and predict, we have to create them with the same signature.
        All the fields that are not in them but we need, we will create as a property of an obj in our class.
        These fields: epochs,batch_size, verbose

        The training is like this:
        1. splits the data to batch_size rows in each chunk.
        2. take a chunk.
        3. forward the data to predict the d+1 return.
        4. backpropagation.
        5. does this epochs times.
        :param num_of_fields: The number of fields.
        :param number_of_layers: The number of layers we want the NN to have.
        :param number_of_neurons_in_the_first_layer: The number of neurons in the first layer.
        :param activation_function: The activation function we want.
        :param probability_flag: A flag: 1 ->, 0->
        :param dived_num_of_neurons_by_each_layer:
                neurons_layer_x-1 = (#neurons_layer_x)/dived_num_of_neurons_by_each_layer
        :param epochs: The num of iteration the models does on all the data
        :param batch_size: The number of rows in each iteration.
        :param verbose: A field for how much does the model speak during the train, 0 -> no output during
        """


        # this is a field because in the Sequential function, there is no field for it
        self.dived_num_of_neurons_by_each_layer = dived_num_of_neurons_by_each_layer

        # these are fields because the fit function we inherit doesn't have them
        if (epochs <= 0):
            raise ValueError('epochs must be greater than 0.')
        self.epochs = epochs

        if (batch_size <= 0):
            raise ValueError('batch_size must be greater than 0.')
        self.batch_size = batch_size

        if(verbose not in [0, 1, 2]):
            raise ValueError('verbose must be 0, 1, or 2.')
        self.verbose = verbose

        if (number_of_neurons_in_the_first_layer is None) or (number_of_neurons_in_the_first_layer <= 0):
            raise ValueError ("num of neurons in first layer must be positive")

        if (probability_flag not in [0, 1]):
            raise ValueError ("probability flag must be 0 or 1")

        max_num_of_layers = self.check_max_num_of_layers(number_of_neurons_in_the_first_layer,
                                                         dived_num_of_neurons_by_each_layer)

        if ((number_of_layers is None) or
            (number_of_layers == 0) or # because of the 'Funnel' dived by 2 approach
            (number_of_layers > max_num_of_layers)):
                raise ValueError ("number of layers is illegal")

        if (dived_num_of_neurons_by_each_layer <= 0):
            raise ValueError ("dived_num_of_neurons_by_each_layer can't be none positive")


        # neurons_layer_x-1 = (#neurons_layer_x)/2
        if (num_of_fields is None or num_of_fields <= 0):
            raise ValueError("num_of_fields must be a number over 0.")

        # input layer
        layers_list = [Input(shape = (num_of_fields,))]

        # adding the layers to the model
        # we already checked max_num_of_layers > number_of_layers
        for i in range(number_of_layers):
            # must be an int
            num_of_neurons = int(number_of_neurons_in_the_first_layer// pow(dived_num_of_neurons_by_each_layer,i))
            if (num_of_neurons < 2):
                break
            layers_list.append(Dense(units = num_of_neurons,
                                     activation=activation_function))

        # output layer
        # if probability_flag == 0 -> activation_function_output is None -> continues value
        # loss function is -> MSE
        # if probability_flag == 1 -> activation_function_output is sigmoid
        # and we need to change the loss function as well -> 'binary_crossentropy
        activation_function_output = None
        # loss function based on the output layer
        loss_function = 'mse'
        if (probability_flag == 1):
            activation_function_output = 'sigmoid'
            loss_function = 'binary_crossentropy'

        output_layer = Dense(1, activation=activation_function_output)
        layers_list.append(output_layer)

        # saving the model we created as a properties of the FCNN object
        self.model = Sequential(layers_list)


        # the model itself has properties, the function for the back-proprgstion,
        # loss function
        self.model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])


    def check_max_num_of_layers(self,number_of_neurons_in_the_first_layer,dived_num_of_neurons_by_each_layer) -> int:
        """
        Calculates the max number of layers of the neural network. Funnel.
        :param number_of_neurons_in_the_first_layer:
        :param dived_num_of_neurons_by_each_layer:
        :return: max number of layers of the neural network we can create.
        """
        start_num_of_neurons = number_of_neurons_in_the_first_layer
        divider = dived_num_of_neurons_by_each_layer
        # cap the length of the model
        if divider <= 1:
            return 100

        counter = 0

        while start_num_of_neurons > 2:
            counter += 1
            start_num_of_neurons = int(start_num_of_neurons // divider)

        max_num_of_layers = counter
        return max_num_of_layers


    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the model on the training data.
        :param x_train: The training set independent fields.
        :param y_train: The training set dependent fields.
        :return: Nothing.
        """
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)


    def predict(self, x_test: pd.DataFrame, train_set: pd.DataFrame | None = None,
                val_set: pd.DataFrame | None = None) -> np.ndarray:
                """
                Predicts the values of the x_test numbers.
                :param x_test: The test set independent fields.
                :param train_set: The training set.
                :param val_set: The validation set.
                :return: The predicted values of the x_test numbers.
                """
                y_pred = self.model.predict(x_test)
                return np.asarray(y_pred).flatten()


