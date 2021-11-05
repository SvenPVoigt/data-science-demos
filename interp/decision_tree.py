import numpy as np
import matplotlib.pyplot as plt

from helpers import *


class Node:
    split: float
    left: "Node" = None
    right: "Node" = None
    leaf: bool = False
    val: float


class DecisionTree:
    def __init__(self, maxLeafSize=1):
        self.root = Node()
        self.maxLeafSize = maxLeafSize


    def fit(self, X, y, nn=None):
        if nn is None:
            nn = self.root

        if len(X) <= self.maxLeafSize:
            nn.leaf = True
            nn.val = np.mean(y)
            return

        nn.left = Node()
        nn.right = Node()

        # Iter over X rows
        split = X[1]
        bestSplit = split
        error = np.sqrt( sum( (y[X<split] - np.mean( y[X<split] ))**2 ) +
            sum( (y[X>=split] - np.mean( y[X>=split] ))**2 ) )
        bestError = error

        for split in X[2:]:
            error = np.sqrt( sum( (y[X<split] - np.mean( y[X<split] ))**2 ) +
                sum( (y[X>=split] - np.mean( y[X>=split] ))**2 ) )

            if error < bestError:
                bestSplit = split
                bestError = error

        nn.split = bestSplit

        self.fit(X[X<bestSplit], y[X<bestSplit], nn.left)
        self.fit(X[X>=bestSplit], y[X>=bestSplit], nn.right)


    def predict(self, x, nn=None):
        if nn is None:
            nn = self.root

        if nn.leaf:
            return nn.val

        if x<nn.split:
            return self.predict(x, nn.left)

        return self.predict(x, nn.right)


def main():
    # load in data per usual
    X_train, y_train = load_data('interp_train.hdf5')
    X_train = X_train.ravel()

    print(X_train.shape, y_train.shape)


    dt = DecisionTree(1)
    dt.fit(X_train, y_train)

    # now load and our testing set for comparison
    X_test, y_test = load_data('interp_test.hdf5')
    X_test = X_test.ravel()

    y_pred = [dt.predict(Xi) for Xi in X_test]

    dt_score = eval_fit(y_pred, y_test)

    print(f"Decision Tree MAE = {dt_score}")

    # Predict for some uniform x to show our fit
    X_plot = np.linspace(min(X_test), max(X_test), 1000)
    y_plot_pred = [dt.predict(Xi) for Xi in X_plot.ravel()]

    # plot the training data points
    plt.scatter(X_train, y_train, marker='.', label="Training samples")
    # plot the testing data points as well
    plt.scatter(X_test, y_test, marker='.', label='Testing samples')

    # plot our fit
    plt.plot(X_plot, y_plot_pred, color='black', linestyle='-', label="Decision Tree fit")

    # set up the plot to make it look nice
    plt.xlabel("Date")
    plt.ylabel("CO2 concentration", rotation=0, horizontalalignment='right')
    plt.title("Decision Tree regression for CO2 interpolation")
    plt.legend()
    plt.tight_layout()
    # show the plot
    plt.show()


if __name__=="__main__":
    print("Running Decision Tree")
    main()