import pandas as pd

import numpy as np
import sklearn.model_selection
import h5py
import matplotlib.pyplot as plt

co2_data = pd.read_csv('../co2_mm_mlo.csv', index_col = False)

# get points between 1970 and 2000
train_points = co2_data[co2_data['decdate'].between(1970, 2000)]
# not between
test_points = co2_data[~co2_data['decdate'].between(1970, 2000)]

print(len(train_points), len(co2_data))

X_train = train_points['decdate']
y_train = train_points['average']

X_test = test_points['decdate']
y_test = test_points['average']

train_file = h5py.File("extrap_train.hdf5", 'w')
train_file.create_dataset("X", data=X_train)
train_file.create_dataset("y", data=y_train)

test_file = h5py.File("extrap_test.hdf5", 'w')
test_file.create_dataset("X", data=X_test)
test_file.create_dataset("y", data=y_test)

# plot the training data points
plt.scatter(X_train, y_train, marker='.', label="Training samples")
# plot the testing data points as well
plt.scatter(X_test, y_test, marker='.', label='Testing samples')

# set up the plot to make it look nice
plt.xlabel("Date")
plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
plt.title("Train/test split for CO2 extrapolation")
plt.legend()
plt.tight_layout()
# show the plot
plt.show()

