import pandas as pd
import numpy as np

dataset = pd.read_csv('R_four_states.csv')
test_set_f = np.array(dataset.iloc[-16:310, 1:2])

# making date as pandas date and setting as a index
dataset['date'] = pd.to_datetime(dataset['date'])
dataset.set_index('date', inplace=True)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
