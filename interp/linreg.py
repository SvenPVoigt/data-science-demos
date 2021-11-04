import h5py
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler # used for color palettes

#matplotlib.rcParams.update({"axes.prop_cycle": cycler('color', [ '#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499'])})

train_color = None #'#a288e3'
test_color = None #'#96a13a'


def load_data(fname):
    # load in an hdf5 file and return the X and y values
    data_file = h5py.File(fname) 

    # load in X and y training data, fully into memory
    X = data_file['X'][:]
    y = data_file['y'][:]
    return X, y

def fit_linear_model(X_train, y_train):
    # each row is a datapoint, each col is a power of x
    X_arr = np.stack((np.ones_like(X_train), X_train), axis=-1)
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
    X_arr = np.stack((np.ones_like(X), X), axis=-1)
    y_pred = np.matmul(X_arr, beta)

    return y_pred

# pull in training data
X_train, y_train = load_data('interp_train.hdf5')

beta = fit_linear_model(X_train, y_train)

print(f"Learned fit is: {beta}")

# Predict for some uniform x to show our fit
X_samp = np.linspace(min(X_train), max(X_train))
y_samp_pred = linear_model_predict(X_samp, beta)

# plot the training data points
plt.scatter(X_train, y_train, color = train_color, marker='.', label="Training samples")

# now load and plot our testing set for comparison
X_test, y_test = load_data('interp_test.hdf5')
plt.scatter(X_test, y_test, color = test_color, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_samp, y_samp_pred, color='black', linestyle='-', label="Linear fit")

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Linear regression for CO2 interpolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()
