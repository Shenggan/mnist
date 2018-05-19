import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from config import *
from sklearn.decomposition import PCA


def read_data(dim=50,reduction=True):
    X_train = np.fromfile(TRAIN_DATA,dtype=np.uint8).reshape(TRAIN_NUM,FIG_W,FIG_W)
    Y_train = np.fromfile(TRAIN_LABEL,dtype=np.uint8)
    X_test = np.fromfile(TEST_DATA,dtype=np.uint8).reshape(TEST_NUM,FIG_W,FIG_W)
    Y_test = np.fromfile(TEST_LABEL,dtype=np.uint8)
    return X_train, Y_train, X_test, Y_test

def pca_curve(X):
    pca = PCA(n_components=10)
    X_r = pca.fit(X).transform(X)
    plt.plot(range(10), pca.explained_variance_ratio_)
    plt.plot(range(10), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Component-wise and Cumulative Explained Variance")
    plt.xlabel("First K Principal Component")
    plt.ylabel("explained portion")
    plt.savefig("hello.png")

def pca_picture(X,y):
    X=X.reshape((-1,2025))
    pca = PCA(n_components=100)
    X_r = pca.fit(X).transform(X)
    plt.scatter(X_r[:1000,0],X_r[:1000,1],c=y[:1000])
    plt.savefig("hello.png")

def visualize(data):
    import random
    ind = random.randint(0,TRAIN_NUM)
    plt.imshow(data[ind],"gray")
    plt.show()

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()
    #pca_curve(X_train[100])
    pca_picture(X_train,Y_train)
    #visualize(X_train)
