"""The utils for ML parts."""

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from config import *


def read_data(dim=50, reduction=True):
    """Read dataset from file

    Read Dataset from file and reduce the dimention using PCA,
    then return the numpy array of trainset and testset.

    Arg:
        dim : the dimention the PCA reduce to.
        reduction : Whether using the PCA to reduce the dimention.

    Returns:
        X_train : The data for trainset
        Y_train : The label for trainset
        X_test : The data for testset
        Y_test : The label for testset
    """
    X_train = np.fromfile(TRAIN_DATA, dtype=np.uint8).reshape(
        TRAIN_NUM, FIG_W, FIG_W)/255.0
    Y_train = np.fromfile(TRAIN_LABEL, dtype=np.uint8)
    X_test = np.fromfile(TEST_DATA, dtype=np.uint8).reshape(
        TEST_NUM, FIG_W, FIG_W)/255.0
    Y_test = np.fromfile(TEST_LABEL, dtype=np.uint8)

    X_train = X_train.reshape((TRAIN_NUM, -1))
    Y_train = Y_train.reshape((TRAIN_NUM, -1))
    X_test = X_test.reshape((TEST_NUM, -1))
    Y_test = Y_test.reshape((TEST_NUM, -1))

    if reduction:
        pca = PCA(n_components=dim)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, Y_train, X_test, Y_test


def visualize(data):
    """Visualization the data

    Random choose a example of the dataset for visualization,
    note it is not recommend use if the dataset has been reduced dimention.

    Arg:
        data : The dataset for visualization.
    """
    ind = random.randint(0, TRAIN_NUM)
    plt.imshow(data[ind], "gray")
    plt.show()


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()

    # Visualization
    # plt.hist(Y_test,bins=10)
    # plt.xlabel("Digits")
    # plt.ylabel("Frequency")
    # plt.savefig("data_dist.png")
    # visualize(X_train)
