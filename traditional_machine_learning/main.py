"""The main scripts for Traditioanl Machine Learinng"""

import sklearn
from sklearn import metrics, svm
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC, NuSVC

from config import *
from utils import *


def argparser():
    """The argumentation parser

    Returns:
        args : The parsed args
    """
    parser = argparse.ArgumentParser(
        description='Traditioanl Machine Learinng Training Argumentation')
    parser.add_argument('--model_type', type=str,
                        choices=['svm', 'nusvm', 'knn', 'lr'], default="svm")
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--dim', type=int, default=100)
    args = parser.parse_args()
    return args


def SVM(X_train, Y_train, X_test, Y_test):
    """SVM Method

    Use OneVsRestClassifier for this multi-class problem.
    And will generate the report for SVM

    Arg:
        X_train : The data for trainset
        Y_train : The label for trainset
        X_test : The data for testset
        Y_test : The label for testset
    """
    classifier = OneVsRestClassifier(SVC(
        C=25, kernel='rbf', gamma=0.05, cache_size=4000, probability=False), n_jobs=24)
    classifier.fit(X_train, Y_train)

    predicted = classifier.predict(X_train)
    print("Train report of SVM ======= ")
    print(metrics.classification_report(Y_train, predicted))

    predicted = classifier.predict(X_test)
    print("Test report of SVM ======= ")
    print(metrics.classification_report(Y_test, predicted))


def NuSVM(X_train, Y_train, X_test, Y_test):
    """NuSVM Method

    Use OneVsRestClassifier for this multi-class problem.
    And will generate the report for NuSVM

    Arg:
        X_train : The data for trainset
        Y_train : The label for trainset
        X_test : The data for testset
        Y_test : The label for testset
    """
    parameters = {'nu': (0.05, 0.02), 'gamma': [3e-2, 2e-2, 1e-2]}
    svc_clf = NuSVC(nu=0.1, kernel='rbf', verbose=False)
    gs_clf = GridSearchCV(svc_clf, parameters, verbose=False, n_jobs=24)
    svc_clf.fit(X_train, Y_train)

    predicted = svc_clf.predict(X_train)
    print("Train report of NuSVM ======= ")
    print(metrics.classification_report(Y_train, predicted))

    predicted = svc_clf.predict(X_test)
    print("Test report of NuSVM ======= ")
    print(metrics.classification_report(Y_test, predicted))


def KNN(X_train, Y_train, X_test, Y_test):
    """KNN Method

    Use OneVsRestClassifier for this multi-class problem.
    And will generate the report for KNN

    Arg:
        X_train : The data for trainset
        Y_train : The label for trainset
        X_test : The data for testset
        Y_test : The label for testset
    """
    knn_clf = KNeighborsClassifier(
        n_neighbors=2, algorithm='kd_tree', weights='distance', p=3, n_jobs=24)
    knn_clf.fit(X_train, Y_train)

    predicted = knn_clf.predict(X_train)
    print("Train report of KNN ======= ")
    print(metrics.classification_report(Y_train, predicted))

    predicted = knn_clf.predict(X_test)
    print("Test report of KNN ======= ")
    print(metrics.classification_report(Y_test, predicted))


def LR(X_train, Y_train, X_test, Y_test):
    """Logist Regression Method

    Use OneVsRestClassifier for this multi-class problem.
    And will generate the report for LR

    Arg:
        X_train : The data for trainset
        Y_train : The label for trainset
        X_test : The data for testset
        Y_test : The label for testset
    """
    lr_clf = LogisticRegression(solver='lbfgs')
    lr_clf.fit(X_train, Y_train)

    predicted = lr_clf.predict(X_train)
    print("Train report of LR ======= ")
    print(metrics.classification_report(Y_train, predicted))

    predicted = lr_clf.predict(X_test)
    print("Test report of LR ======= ")
    print(metrics.classification_report(Y_test, predicted))


def main(args):
    model_type = args.model_type

    train_size_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if args.pca:
        X_train, Y_train, X_test, Y_test = read_data()
    else
        X_train, Y_train, X_test, Y_test = read_data(args.dim)

    for ratio in train_size_list:
        print("Train with %d%% training set and test on the whole test set begins" % (
            100-100*portion))
        X_train_temp, _, Y_train_temp, _ = train_test_split(
            X_train, Y_train, test_size=portion)
        if model_type == 'svm':
            SVM(X_train_temp, Y_train_temp, X_test, Y_test)
        if model_type == 'nusvm':
            NuSVM(X_train_temp, Y_train_temp, X_test, Y_test)
        if model_type == 'knn':
            KNN(X_train_temp, Y_train_temp, X_test, Y_test)
        if model_type == 'lr':
            LR(X_train_temp, Y_train_temp, X_test, Y_test)


if __name__ == "__main__":
    args = argparser()
    main(args)
