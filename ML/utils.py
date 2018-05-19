import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
from config import *
#import seaborn as sns
from sklearn.decomposition import PCA


def read_data(dim=50,reduction=True):
    X_train = np.fromfile(TRAIN_DATA,dtype=np.uint8).reshape(TRAIN_NUM,FIG_W,FIG_W)/255.0
    Y_train = np.fromfile(TRAIN_LABEL,dtype=np.uint8)
    X_test = np.fromfile(TEST_DATA,dtype=np.uint8).reshape(TEST_NUM,FIG_W,FIG_W)/255.0
    Y_test = np.fromfile(TEST_LABEL,dtype=np.uint8)
    X_train = X_train.reshape((TRAIN_NUM,-1))
    Y_train = Y_train.reshape((TRAIN_NUM,-1))
    X_test = X_test.reshape((TEST_NUM,-1))
    Y_test = Y_test.reshape((TEST_NUM,-1))
    if (reduction):
        pca = PCA(n_components=dim)
        pca.fit(X_train)
        X_train = pca.transform(X_train) 
        X_test = pca.transform(X_test)
    return X_train, Y_train, X_test, Y_test


def visualize(data):
    import random
    ind = random.randint(0,TRAIN_NUM)
    plt.imshow(data[ind],"gray")
    plt.show()

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()
    """
    plt.hist(Y_test,bins=10)
    plt.xlabel("Digits")
    plt.ylabel("Frequency")
    plt.savefig("data_dist.png")
    """
    #visualize(X_train)
    
    
