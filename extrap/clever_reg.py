import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from scipy.optimize import curve_fit

# load in data per usual
X_train, y_train = load_data('extrap_train.hdf5')

def func(x, A, B, C, D, E, alpha, phi, x0):
    x = x - x0
    y = A + B * x + C*x**2 + D * np.cos(2 * np.pi * x + phi) #+ E * np.exp(x / alpha)
    return y.squeeze()

p0 = [10, 10, 10, 10, 10, 50, 0, 2000]
popt, pcov = curve_fit(func, X_train, y_train.squeeze(), p0 = p0)

print(popt)

# now make a wrapper for our fit function
clever_model = lambda x : func(x, *popt)

# now load and our testing set for comparison
X_test, y_test = load_data('extrap_test.hdf5')

clever_score = eval_fit(clever_model(X_test), y_test)

print(f"Exp-sinusoid MAE = {clever_score}")

# Predict for some uniform x to show our fit
X_plot = np.linspace(min(X_test), max(X_test), 1000)
y_plot_pred = clever_model(X_plot)

# plot the training data points
plt.scatter(X_train, y_train, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_plot, y_plot_pred, color='black', linestyle='-', label="Exp-sinusoid fit")

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Polynomial regression for CO2 extrapolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()
