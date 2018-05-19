import sklearn
from sklearn import svm, metrics
from sklearn.svm import SVC,NuSVC
from sklearn.grid_search import   GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC,LinearSVC
from utils import *
from config import *
#import time
from sklearn.preprocessing import MinMaxScaler

def SVM(X_train, Y_train, X_test, Y_test):
    classifier = OneVsRestClassifier(SVC(C=25,kernel='rbf',gamma=0.05,cache_size=4000,probability=False),n_jobs=24)
    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_train)
    print ("Train report of SVM ======= ")
    print (metrics.classification_report(Y_train, predicted))
    predicted = classifier.predict(X_test)
    print ("Test report of SVM ======= ")
    print (metrics.classification_report(Y_test, predicted))

def NuSVM(X_train, Y_train, X_test, Y_test):
    parameters = {'nu':(0.05, 0.02) , 'gamma':[3e-2, 2e-2, 1e-2]}
    svc_clf=NuSVC(nu=0.1, kernel='rbf', verbose=False )
    gs_clf =  GridSearchCV(svc_clf, parameters, verbose=False, n_jobs=24)
    svc_clf.fit(X_train, Y_train)
    predicted = svc_clf.predict(X_train)
    print ("Train report of SVM ======= ")
    print (metrics.classification_report(Y_train, predicted))
    predicted = svc_clf.predict(X_test)
    print ("Test report of SVM ======= ")
    print (metrics.classification_report(Y_test, predicted))

def KNN(X_train, Y_train, X_test, Y_test):
    knn_clf=KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree', weights='distance', p=3, n_jobs=24)
    knn_clf.fit(X_train, Y_train)
    predicted = knn_clf.predict(X_train)
    print ("Train report of KNN ======= ")
    print (metrics.classification_report(Y_train, predicted))
    predicted = knn_clf.predict(X_test)
    print ("Test report of KNN ======= ")
    print (metrics.classification_report(Y_test, predicted))

def LR(X_train, Y_train, X_test, Y_test):
#    lr_clf=LogisticRegression(solver ='lbfgs', multi_class='multinomial', max_iter=1000,  C=1e-5, n_jobs=4)
    lr_clf=LogisticRegression(solver ='lbfgs')
#    parameters = { 'C':[1e-5,1e-2,10]}
#    gs_clf =  GridSearchCV(lr_clf, parameters, n_jobs=10, verbose=False )
    lr_clf.fit(X_train,Y_train)
    predicted = lr_clf.predict(X_train)
    print ("Train report of LR ======= ")
    print (metrics.classification_report(Y_train, predicted))
    predicted = lr_clf.predict(X_test)
    print ("Test report of LR ======= ")
    print (metrics.classification_report(Y_test, predicted))

if __name__ == "__main__":
    """
    X_train, Y_train, X_test, Y_test = read_data()
    for portion in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print ("Train with %d%% training set and test on the whole test set begins"%(100-100*portion))
        X_train_temp , _, Y_train_temp, _ = train_test_split( \
                X_train, Y_train, test_size=portion, random_state=42)
        SVM(X_train_temp, Y_train_temp, X_test, Y_test)
        #NuSVM(X_train_temp, Y_train_temp, X_test, Y_test)
    """
    X_train, Y_train, X_test, Y_test = read_data(20)
    for portion in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print ("Train with %d%% training set and test on the whole test set begins"%(100-100*portion))
        X_train_temp , _, Y_train_temp, _ = train_test_split( \
                X_train, Y_train, test_size=portion, random_state=42)
        KNN(X_train_temp, Y_train_temp, X_test, Y_test)
    """
    X_train, Y_train, X_test, Y_test = read_data(100)
    for portion in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print ("Train with %d%% training set and test on the whole test set begins"%(100-100*portion))
        X_train_temp , _, Y_train_temp, _ = train_test_split( \
                X_train, Y_train, test_size=portion, random_state=42)
        LR(X_train_temp, Y_train_temp, X_test, Y_test)
    """
