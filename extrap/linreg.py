import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib

from helpers import *

def fit_linear_model(X_train, y_train):
    # each row is a datapoint, each col is a power of x
    X_arr = np.concatenate((np.ones_like(X_train), X_train), axis=-1)
    y_arr = y_train # just use the training y vals
    print(X_arr.shape, y_arr.shape)

    ## NOTE: the following is generally a very bad idea! I am only demonstrating it for educational purposes!
    # DO NOT FORM THIS MATRIX EXPLICITLY!!!
    xtx = np.matmul(X_arr.T, X_arr)
    xtxinv = np.linalg.inv(xtx)
    xty = np.matmul(X_arr.T, y_arr)
    beta = np.matmul(xtxinv, xty)

    return beta

def linear_model_predict(X, beta):
    X_arr = np.concatenate((np.ones_like(X), X), axis=-1)
    y_pred = np.matmul(X_arr, beta)

    return y_pred



# pull in training data
X_train, y_train = load_data('extrap_train.hdf5')

beta = fit_linear_model(X_train, y_train)
print(f"Learned fit is: {beta}")

# also train a regression model using sklearn
skl_model = LinearRegression()
skl_model.fit(X_train, y_train)
print("Sklearn fit is:", skl_model.intercept_, skl_model.coef_)

X_test, y_test = load_data('extrap_test.hdf5')


# Predict for some uniform x to show our fit
X_plot = np.linspace(min(X_test), max(X_test), 1000)
y_plot_pred = linear_model_predict(X_plot, beta)

# also fit with sklearn to show that they are the same
y_plot_skl = skl_model.predict(X_plot)


# now load and our testing set for comparison

skl_linreg_score = eval_fit(skl_model.predict(X_test), y_test)

print(f"Linear fit MAE = {skl_linreg_score}")

# plot the training data points
plt.scatter(X_train, y_train, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_plot, y_plot_pred, color='black', linestyle='--', label="Normal EQ Linear fit")
plt.plot(X_plot, y_plot_skl, color='green', linestyle='-', label="SKlearn Linear fit")

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Linear regression for CO2 extrapolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()
