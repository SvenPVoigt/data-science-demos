import pandas as pd

import numpy as np
import sklearn.model_selection
import h5py

co2_data = pd.read_csv('../co2_mm_mlo.csv', index_col = False)

num_points = len(co2_data)

[train_inds, test_inds] = sklearn.model_selection.train_test_split(np.arange(num_points), train_size = 0.5, random_state = 7)
# sort them for convenience later
train_inds = sorted(train_inds)
test_inds = sorted(test_inds)

train_file = h5py.File("interp_train.hdf5", 'w')
train_file.create_dataset("X", data=co2_data['decdate'][train_inds])
train_file.create_dataset("y", data=co2_data['average'][train_inds])

test_file = h5py.File("interp_test.hdf5", 'w')
test_file.create_dataset("X", data=co2_data['decdate'][test_inds])
test_file.create_dataset("y", data=co2_data['average'][test_inds])



