"""The visulization scripts for PCA"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from config import *
from utils import read_data

plt.switch_backend('agg')


def pca_curve(X):
    """Component-wise and Cumulative Explained Variance 

    Visulization for Component-wise and Cumulative Explained Variance.
    This function will plot a figure that 
        xlabel : First K Principal Component
        ylabel : Explained Variance Ratio

    Arg:
        X : The data of dataset for visualization.
    """
    pca = PCA(n_components=10)
    X_r = pca.fit(X).transform(X)

    plt.plot(range(10), pca.explained_variance_ratio_)
    plt.plot(range(10), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Component-wise and Cumulative Explained Variance")
    plt.xlabel("First K Principal Component")
    plt.ylabel("explained portion")
    plt.savefig("pca_1.png")


def pca_picture(X, y):
    def pca_curve(X):
    """Component-wise and Cumulative Explained Variance 

    Visulization for Component-wise and Cumulative Explained Variance.
    This function will plot a figure that 
        xlabel : First K Principal Component
        ylabel : Explained Variance Ratio

    Arg:
        X : The data of dataset for visualization.
        y : The label of dataset for visualization.
    """
    X = X.reshape((-1, 2025))
    pca = PCA(n_components=100)
    X_r = pca.fit(X).transform(X)
    plt.scatter(X_r[:1000, 0], X_r[:1000, 1], c=y[:1000])
    plt.savefig("pca_2.png")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()
    pca_picture(X_train, Y_train)
    # pca_curve(X_train[100])
