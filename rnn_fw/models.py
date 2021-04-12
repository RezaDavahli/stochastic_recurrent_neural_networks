import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import copy
import seaborn as sns

# scaling
sc = MinMaxScaler(feature_range=(0, 1))


# Multivariate Guassian kernel
def calc_pdf(y, mu, var):
    value = tf.subtract(y, mu) ** 2
    value = (1 / tf.math.sqrt(2 * np.pi * var)) * tf.math.exp((-1 / (2 * var)) * value)
    return value


# Error function in terms of negative logarithm likelihood
def mdn_loss(y_true, pi, mu, var):
    out = calc_pdf(y_true, mu, var)
    # multiply with each pi and sum it
    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims=True)
    out = -tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)


def training(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    # model.train()
    with tf.GradientTape() as tape:
        pi_, mu_, var_ = model(train_x, training=True)
        # calculate loss
        loss = mdn_loss(train_y, pi_, mu_, var_)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def validation(model, optimizer, train_x, train_y):
    # GradientTape: Trace operations to compute gradients
    # model.eval()
    with tf.GradientTape() as tape:
        pi_, mu_, var_ = model(train_x, training=True)
        # calculate loss
        loss = mdn_loss(train_y, pi_, mu_, var_)
    # compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Creating a sample of a number of points from the distribution and
# generate a dense set of predictions instead of picking just one.
def sample_predictions(pi_vals, mu_vals, var_vals, samples=1):
    n, k = pi_vals.shape
    l = 1
    print('shape: ', n, k, l)
    # place holder to store the y value for each sample of each row
    out = np.zeros((n, samples, l))
    for i in range(n):
        for j in range(samples):
            # for each sample, use pi/probs to sample the index
            # that will be used to pick up the mu and var values
            idx = np.random.choice(range(k), p=pi_vals[i])
            for li in range(l):
                # Draw random sample from gaussian distribution
                out[i, j, li] = np.random.normal(mu_vals[i, idx * (li + l)], np.sqrt(var_vals[i, idx]))
    return out


# Testing for stationary and non-stationary
def stationary_test(data):
    i = 'CALIFORNIA'
    dftest = adfuller(data[i], autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


# Removing seasonality and trend
def removing_seasonality_trend(dataset):
    # 1- seasonality
    order = 2
    coef = np.polyfit(np.arange(len(dataset['CALIFORNIA'])),
                      dataset['CALIFORNIA'].values.ravel(),
                      order)
    poly_mdl = np.poly1d(coef)
    poly_mdl
    trend = pd.Series(data=poly_mdl(np.arange(len(dataset['CALIFORNIA']))),
                      index=dataset.index)
    detrended = dataset['CALIFORNIA'] - trend
    seasonal = detrended.groupby(by=detrended.index.month).mean()
    col = 'CALIFORNIA'
    seasonal_component = copy.deepcopy(dataset)
    for i in seasonal.index:
        seasonal_component.loc[seasonal_component.index.month == i, col] = seasonal.loc[i]
    deseasonal = dataset - seasonal_component

    # 2- Removing trend
    coef = np.polyfit(np.arange(len(deseasonal)), deseasonal['CALIFORNIA'], order)
    poly_mdl = np.poly1d(coef)
    trend_comp = pd.DataFrame(data=poly_mdl(np.arange(len(dataset['CALIFORNIA']))),
                              index=dataset.index,
                              columns=['CALIFORNIA'])

    residual = dataset - seasonal_component - trend_comp
    trend_comp_test = trend_comp.iloc[-16:310, 0:1]
    seasonal_component_test = seasonal_component.iloc[-16:310, 0:1]
    print(seasonal_component_test.shape)
    return residual, trend_comp_test, seasonal_component_test


# Visualize learning
def plot_losses(training_losses, validation_losses):
    fig, axs = plt.subplots(ncols=2)
    for num, losses in enumerate[training_losses, validation_losses]:
        losses_float = [float(loss_.cpu().detach().numpy()) for loss_ in losses]
        loss_indices = [i for i, l in enumerate(losses_float)]
        axs[num] = sns.lineplot(loss_indices, losses_float)
    plt.savefig(f"Fold__losses.jpg")


def splitting(data, percentage):
    num_data = len(data)
    num_train = int(percentage * num_data)
    first_set = data[0:num_train]
    second_set = data[num_train:]
    return first_set, second_set
