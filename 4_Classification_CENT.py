import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import time

#This script uses the CENT as input features to classify images.
#6 classifiers are tested: Logistic Regression, Naive Bayes, Linear SVM, SVM, Dicision Tree, Random Forest
# Besure to run 3_Exp_Layer_CENT.py before you run this script.

def Dict_To_Array(X_dict, n_classes):
    total_len = 0
    for i in np.arange(n_classes):
        total_len+=len(X_dict[i])
    X = np.zeros((total_len, 2))
    y= np.zeros(total_len)

    index_count = 0
    for i in np.arange(n_classes):
        X[index_count:index_count+len(X_dict[i])] = X_dict[i]
        y[index_count:index_count + len(X_dict[i])] = i
        index_count+=len(X_dict[i])

    return X, y

def Experiment_LogisticRegression(X_tr, y_tr, X_te, y_te):
    clf = linear_model.LogisticRegression()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred == y_te) / 1.0 / len(y_te)
    print('Linear Regression testing finished!')
    print('Linear Regression testing accuracy:', acc)

    return acc

def Experiment_NaiveBayes(X_tr, y_tr, X_te, y_te):
    clf = GaussianNB()
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred == y_te) / 1.0 / len(y_te)
    print('Linear Regression testing finished!')
    print('Linear Regression testing accuracy:', acc)

    return acc

def Experiment_LinearSVM(X_tr, y_tr, X_te, y_te):
    clf = LinearSVC(random_state=0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred==y_te)/1.0/len(y_te)
    print('Linear SVM testing finished!')
    print('Linear SVM testing accuracy:', acc)

    return acc

def Experiment_SVM(X_tr, y_tr, X_te, y_te):
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    start_time = time.time()
    clf = svm.SVC(random_state=0, kernel='rbf', max_iter=-1)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred==y_te)/1.0/len(y_te)
    print('SVM testing finished!')
    print('SVM testing accuracy:', acc, '     training duration:', time.time()-start_time, ' s')

    return acc

def Experiment_DecisionTree(X_tr, y_tr, X_te, y_te):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred == y_te) / 1.0 / len(y_te)
    print('Decision Tree testing finished!')
    print('Decision Tree testing accuracy:', acc)

    return acc

def Experiment_RandomForest(X_tr, y_tr, X_te, y_te):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = np.sum(y_pred == y_te) / 1.0 / len(y_te)
    print('Random Forrest testing finished!')
    print('Random Forrest testing accuracy:', acc)

    return acc

def main():
    n_class = 10
    train_model = True
    plot = True

    f1 = open("layer_output_dict/layer_CENT_train.pkl", 'rb')
    layer_CENT_train = pkl.load(f1)
    f1.close()

    f2 = open("layer_output_dict/layer_CENT_test.pkl", 'rb')
    layer_CENT_test = pkl.load(f2)
    f2.close()

    acc_array = np.zeros((6,9))
    for i in np.arange(9):
        n_class = i+2
        X_tr, y_tr = Dict_To_Array(layer_CENT_train,n_class)
        X_te, y_te = Dict_To_Array(layer_CENT_test, n_class)

        acc_array[0, i] = Experiment_LogisticRegression(X_tr, y_tr, X_te, y_te)
        acc_array[1, i] = Experiment_NaiveBayes(X_tr, y_tr, X_te, y_te)
        acc_array[2,i] = Experiment_LinearSVM(X_tr, y_tr, X_te, y_te)
        acc_array[3,i] = Experiment_SVM(X_tr, y_tr,X_te, y_te)
        acc_array[4,i] = Experiment_DecisionTree(X_tr, y_tr,X_te, y_te)
        acc_array[5,i] = Experiment_RandomForest(X_tr, y_tr,X_te, y_te)

    np.save('acc_array', acc_array)
    print('accuracy array saved successfully!')
    return 0

main()
