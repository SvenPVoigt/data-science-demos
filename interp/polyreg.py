import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

from helpers import *

# load the data as before
X_train, y_train = load_data('interp_train.hdf5')

poly_trans = PolynomialFeatures(degree = 3)
X_poly = poly_trans.fit_transform(X_train)

# also train a regression model using sklearn
skl_model = LinearRegression()
skl_model.fit(X_poly, y_train)
print("Sklearn poly fit is:", skl_model.intercept_, skl_model.coef_)

# now load and our testing set for comparison
X_test, y_test = load_data('interp_test.hdf5')

skl_linreg_score = eval_fit(skl_model.predict(poly_trans.transform(X_test)), y_test)

print(f"Polynomial fit MAE = {skl_linreg_score}")


# Predict for some uniform x to show our fit
X_plot = np.linspace(min(X_train), max(X_train), 1000)
y_plot_pred = skl_model.predict(poly_trans.transform(X_plot))

""" SAME PLOTTING CODE AS BEFORE """
# plot the training data points
plt.scatter(X_train, y_train, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, marker='.', label='Testing samples')

# plot our fit
plt.plot(X_plot, y_plot_pred, color='black', linestyle='-', label="3rd-order polynomial fit")

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Polynomial regression for CO2 interpolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()
