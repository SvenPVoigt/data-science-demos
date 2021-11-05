import h5py
import numpy as np

def load_data(fname):
    # load in an hdf5 file and return the X and y values
    data_file = h5py.File(fname) 

    # load in X and y training data, fully into memory
    X = data_file['X'][:].reshape(-1, 1) # each row is a data point
    y = data_file['y'][:]
    return X, y

def eval_fit(y_pred, y_true):
    # compute  mean absolute error
    mae = np.mean(np.abs(y_pred - y_true))
    return mae  # don't normalize