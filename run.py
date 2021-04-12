import rnn_fw as rf
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras import backend as K
from keras.layers import Input
import matplotlib.pyplot as plt

residual, trend_comp_test, seasonal_component_test = rf.models.removing_seasonality_trend(rf.utils.dataset)
training_data = residual.iloc[:, 0:1].values

# Defining training set and test set
num_data = len(training_data)
first_split = 0.95
num_train = int(first_split * num_data)
total_set = training_data[0:num_train]
test_set = training_data[num_train:]
second_split = 0.8
num_data_two = len(total_set)
num_train_two = int(second_split * num_data_two)
training_set = total_set[0:num_train_two]
val_set = training_data[num_train_two:]
print(training_set.shape)
print(val_set.shape)
print(test_set.shape)

# Building the test data
dataset_total = training_data
inputs = dataset_total[len(dataset_total) - len(test_set) - 14:]
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
inputs = rf.models.sc.fit_transform(inputs)
x_test = []
for i in range(14, 30):
    x_test.append(inputs[i - 14:i, 0:1])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
x_test = x_test.astype('float32')

training_set_scaled = rf.models.sc.fit_transform(training_set)
validating_set_scaled = rf.models.sc.fit_transform(val_set)

nfolds = 5
EPOCHS = 2000
MAPE_TOTAL = []
for fold in range(nfolds):
    # creating sequence length
    # For this problem, the model is trained based on data from last 14 days to predict the next day.
    x_train = []
    y_train = []
    for i in range(14, 235):
        x_train.append(training_set_scaled[i - 14:i, 0:1])
        y_train.append(training_set_scaled[i, 0:1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    x_val = []
    y_val = []
    for i in range(14, 75):
        x_val.append(validating_set_scaled[i - 14:i, 0:1])
        y_val.append(validating_set_scaled[i, 0:1])
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))
    x_val = x_val.astype('float32')
    y_val = y_val.astype('float32')

    # building RNN
    inputs1 = Input(shape=(x_train.shape[1], x_train.shape[2]))
    lstm1 = LSTM(10, return_sequences=True, dtype='float32')(inputs1)
    lstm3 = LSTM(10, dtype='float32')(lstm1)
    layer = tf.keras.layers.Dense(10, dtype='float32')(lstm3)
    #  input feature
    l = 1
    # Number of Gaussian to represent the multimodal distribution
    k = 1
    mu = tf.keras.layers.Dense((l * k), activation=None, dtype='float32')(layer)
    var = tf.keras.layers.Dense(k, activation=K.exp, dtype='float32')(layer)
    pi = tf.keras.layers.Dense(k, activation='softmax', name='pi_layer', dtype='float32')(layer)
    model = tf.keras.models.Model(inputs1, [pi, mu, var])
    outputs = [pi, mu, var]
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss={'output': rf.models.mdn_loss})
    model.summary()

    N = x_train.shape[0]
    dataset = tf.data.Dataset \
        .from_tensor_slices((x_train, y_train)) \
        .shuffle(N).batch(N)
    M = x_train.shape[0]
    dataset_valid = tf.data.Dataset \
        .from_tensor_slices((x_val, y_val)) \
        .shuffle(M).batch(M)

    losses_tr = []
    losses_vl = []
    print_every = int(0.2 * EPOCHS)
    # Define model and optimizer
    model = tf.keras.models.Model(inputs1, [pi, mu, var])
    # Start training
    print('Print every {} epochs'.format(print_every))
    for i in range(EPOCHS):
        for train_x, train_y in dataset:
            loss_tr = rf.models.training(model, optimizer, train_x, train_y)
            losses_tr.append(loss_tr)
        if i % print_every == 0:
            print('Epoch {}/{}: loss {}'.format(i, EPOCHS, losses_tr[-1]))
        for val_x, val_y in dataset_valid:
            loss_vl = rf.models.validation(model, optimizer, val_x, val_y)
            losses_vl.append(loss_vl)

    plt.plot(range(len(losses_tr)), losses_tr)
    plt.plot(range(len(losses_vl)), losses_vl)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss & validation loss')
    plt.show()

    pi_vals, mu_vals, var_vals = model.predict(x_test)

    sampled_predictions = rf.models.sample_predictions(pi_vals, mu_vals, var_vals, 1)
    print(sampled_predictions.shape)

    # Building the test data
    dataset_total = training_data
    inputs = dataset_total[len(dataset_total) - len(test_set) - 14:]
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1]))
    inputs = rf.models.sc.fit_transform(inputs)
    x_test = []
    for i in range(14, 30):
        x_test.append(inputs[i - 14:i, 0:1])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_test = x_test.astype('float32')

    predicted_set = sampled_predictions.reshape(sampled_predictions.shape[0], sampled_predictions.shape[2])
    predicted_set = rf.models.sc.inverse_transform(predicted_set)
    print(predicted_set.shape)

    predicted_set_f = []
    trend_comp_test = np.array(trend_comp_test)
    seasonal_component_test = np.array(seasonal_component_test)
    predicted_set_f = np.array([[predicted_set[i][j] + trend_comp_test[i][j] + seasonal_component_test[i][j]
                                 for j in range(len(predicted_set[0]))] for i in range(len(predicted_set))])

    test_dataframe = pd.DataFrame(rf.utils.test_set_f)
    predict_dataframe = pd.DataFrame((predicted_set_f))

    ax = test_dataframe.plot()
    predict_dataframe.plot(ax=ax)
    plt.legend(['actual', 'predict'])
    plt.show()

    MAPE = rf.utils.mean_absolute_percentage_error(y_true=test_dataframe, y_pred=predict_dataframe)
    print(MAPE)
    MAPE_TOTAL.append(MAPE)


print(MAPE_TOTAL)


