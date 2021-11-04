import h5py
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib

train_color = None #'#a288e3'
test_color = None #'#96a13a'


def load_data(fname):
    # load in an hdf5 file and return the X and y values
    data_file = h5py.File(fname) 

    # load in X and y training data, fully into memory
    X = data_file['X'][:].reshape(-1, 1) # each row is a data point
    y = data_file['y'][:]
    return X, y

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

def eval_fit(y_pred, y_true):
    y_avg = np.average(y_true)
    # compute  mean absolute error
    mae = np.mean(np.abs(y_pred - y_true))

    # normalize to get an easier number to interpret
    return mae / y_avg

# pull in training data
X_train, y_train = load_data('interp_train.hdf5')

beta = fit_linear_model(X_train, y_train)
print(f"Learned fit is: {beta}")

# also train a regression model using sklearn
skl_model = LinearRegression()
skl_model.fit(X_train, y_train)
print("Sklearn fit is:", skl_model.intercept_, skl_model.coef_)



# Predict for some uniform x to show our fit
X_samp = np.linspace(min(X_train), max(X_train))
y_samp_pred = linear_model_predict(X_samp, beta)

# also fit with sklearn to show that they are the same
y_samp_skl = skl_model.predict(X_samp)


# now load and our testing set for comparison
X_test, y_test = load_data('interp_test.hdf5')

skl_linreg_score = eval_fit(skl_model.predict(X_test), y_test)

print(f"Linear fit NMAE = {skl_linreg_score}")

# plot the training data points
plt.scatter(X_train, y_train, color = train_color, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, color = test_color, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_samp, y_samp_pred, color='black', linestyle='-', label="Linear fit")
plt.plot(X_samp, y_samp_skl, color='cyan', linestyle='--', label="SKlearn Linear fit")

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Linear regression for CO2 interpolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()
