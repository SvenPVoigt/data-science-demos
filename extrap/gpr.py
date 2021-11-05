from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel, RationalQuadratic

import numpy as np
import matplotlib.pyplot as plt

from helpers import *

# load in data per usual
X_train, y_train = load_data('extrap_train.hdf5')


# create a GP model

# use kernel previously used for this data: https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html
k1 = 66.0 ** 2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = (
    2.4 ** 2
    * RBF(length_scale=90.0)
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)
)  # seasonal component
# medium term irregularity
k3 = 0.66 ** 2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18 ** 2 * RBF(length_scale=0.134) + WhiteKernel(
    noise_level=0.19 ** 2
)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4

gpr = GaussianProcessRegressor(kernel=kernel_gpml, normalize_y=True, alpha=0)

print(X_train.shape)

gpr.fit(X_train, y_train)

X_test, y_test = load_data('extrap_test.hdf5')

# Predict for some uniform x to show our fit
X_plot = np.linspace(min(X_test), max(X_test), 1000)
# GPR predictions take many samples to estimate a mean/std
[y_plot_pred, y_plot_std] = gpr.predict(X_plot, return_std=True)

gpr_score = eval_fit(gpr.predict(X_test), y_test)
print(f"GPR MAE = {gpr_score}")


""" SAME PLOTTING CODE AS BEFORE """
# plot the training data points
plt.scatter(X_train, y_train, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_plot, y_plot_pred, color='black', linestyle='-', label="GPR fit")

# also plot 1 std deviation uncertaint
plt.fill_between(
    X_plot[:, 0], y_plot_pred - y_plot_std, y_plot_pred + y_plot_std, color="green", alpha=0.2, label="GPR 1$\sigma$ bound"
)

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Polynomial regression for CO2 extrapolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()



