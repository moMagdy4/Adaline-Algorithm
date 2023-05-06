
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def Pre_processing(data):
    data['gender'].fillna(data['gender'].mode()[0], inplace=True)
    data['gender'] = data.gender.map(dict(male=0, female=1))
    return data


def label_encoding(class1, class2, data, feature1, feature2):
    # data['species'] = data.species.map(dict(class1=-1, class2=1))
    # data.dropna(inplace=True)
    # Pre_processing(data)
    for i in range(len(data['species'])):
        if (data['species'][i] == class1):
            data.loc[i, 'species'] = -1
        elif (data['species'][i] == class2):
            data.loc[i, 'species'] = 1
        else:
            data.drop(i, axis=0, inplace=True,)
    data.reset_index(drop=True, inplace=True)
    data = pd.DataFrame().assign(
        feature1=data[feature1], feature2=data[feature2], species=data['species'])
    return data


def Split(data):
    X = data.drop('species', axis=1)
    y = data['species']
    # spliting data to train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, train_size=0.6, shuffle=True, random_state=10, stratify=y)

   # converting to numpy
    X_train = x_train.to_numpy()
    X_test = x_test.to_numpy()
    Y_train = y_train.to_numpy()
    Y_test = y_test.to_numpy()

    m = X_train.shape[0]  # number of training examples(60)

    Y_train = Y_train.reshape(m, 1)
    Y_test = Y_test.reshape(X_test.shape[0], 1)

    return X_train, X_test, Y_train, Y_test


def split2(data):
    X1 = data.drop('species', axis=1)
    plo = data.drop('species', axis=1)
    y = data['species']
    X = FeatureScalling(X1)
    plot = FeatureScalling(plo)

    X = X.to_numpy()
    y = y.to_numpy()
    plot = plot.to_numpy()

    index_c1 = list(range(0, 50))
    index_c2 = list(range(50, 100))
    random.shuffle(index_c1)
    random.shuffle(index_c2)

    trian_indices = index_c1[:30]+index_c2[:30]
    test_indices = index_c1[30:]+index_c2[30:]

    random.shuffle(trian_indices)
    random.shuffle(test_indices)

    X_train = X[trian_indices, :]
    Y_train = y[trian_indices]
    X_test = X[test_indices, :]
    Y_test = y[test_indices]

    m = X_train.shape[0]  # number of training examples(60)
    Y_train = Y_train.reshape(m, 1)
    Y_test = Y_test.reshape(X_test.shape[0], 1)
    return X_train, X_test, Y_train, Y_test, plot


def FeatureScalling(X):
    for column in X.columns:
        X[column] = (X[column] - X[column].min()) / \
            (X[column].max() - X[column].min())
    return X


def fit(X_train, Y_train, le, epochs, bias):
    #b = 0
    m = X_train.shape[0]  # number of training examples(60)
    n = X_train.shape[1]  # number of training features(2)
    if bias == False:
        b = 0
    else:
        b = np.random.randn()

    # w=np.random.randn(n,1) #intialize w vector
    #b = np.random.randn()
    w = np.random.randn(n, 1)

    for j in range(epochs):
        for i in range(m):
            pred = np.dot(X_train[i, :].reshape(1, n), w)+b
            pred = signum(pred)
            if (pred != Y_train[i, 0]):
                error = Y_train[i, 0] - pred
                w = w + (le*error*X_train[i, :].reshape(1, n).T)
                if bias == True:
                    b = b + le * error
    return w, b


def fit_adaline_learning(X_train, Y_train, le, epochs, bias, thresh_hold):
    m = X_train.shape[0]  # number of training examples(60)
    n = X_train.shape[1]  # number of training features(2)

    if bias == False:
        b = 0
    else:
        b = np.random.randn()

    w = np.random.randn(n, 1)
    for j in range(epochs):
        # print("epoch")
        # print(j)
        #error = 0
        #Mean_Square_Error = 0

        break1 = 0
        break2 = 0

        for i in range(m):

            pred = np.dot(X_train[i, :].reshape(1, n), w)+b
            error = (Y_train[i, 0] - pred)
            w = w + (le*error*X_train[i, :].reshape(1, n).T)
            if bias == True:
                b = b + le * error

            break1 = break1+1
        print(break1)
        mse = 0
        for z in range(m):
            yi = np.dot(X_train[z, :].reshape(1, n), w)+b
            mse_error = (Y_train[z, 0]-yi)
            mse += mse_error*mse_error
            break2 = break2+1
        print(break2)

        #sq = float(np.square(Y_train[z, 0]-yi))+float(sq)
        Mean_Square_Error = mse/60
        print("MSE")
        # print(Mean_Square_Error)
        # print(np.shape(w))

        if (Mean_Square_Error < thresh_hold):
            break
        # print("MSE")
        # print(Mean_Square_Error)
    return w, b, Mean_Square_Error


def signum(pred):
    if (pred >= 0):
        return 1
    if (pred < 0):
        return -1


def confusionMatrix(w, b, x_test, y_test):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    m = x_test.shape[0]  # number of training examples(40)
    n = x_test.shape[1]  # number of training features(2)

    for i in range(m):
        pred = np.dot(x_test[i, :].reshape(1, n), w) + b
        pred = signum(pred)
        if (pred == y_test[i, 0] and pred == 1):
            truePositive += 1
        elif (pred == y_test[i, 0] and pred == -1):
            trueNegative += 1
        elif (pred != y_test[i, 0] and pred == 1):
            falsePositive += 1
        elif (pred != y_test[i, 0] and pred == -1):
            falseNegative += 1

    accurcy = (truePositive+trueNegative)/(truePositive +
                                           trueNegative+falsePositive+falseNegative)

    return truePositive, trueNegative, falsePositive, falseNegative, accurcy


def Model(data, class_1, class_2, feature_1, feature_2, le, epochs, bias, thresh_hold):
    new_data = Pre_processing(data)
    pre_processed_data = label_encoding(
        class_1, class_2, new_data, feature_1, feature_2)
    #pre_processed_data2 = FeatureScalling(pre_processed_data)

    X_train, X_test, Y_train, Y_test, pp = split2(pre_processed_data)

    weight, b, mse = fit_adaline_learning(
        X_train, Y_train, le, epochs, bias, thresh_hold)

    tp, tn, fp, fn, acc = confusionMatrix(weight, b, X_test, Y_test)

    plot_decision_boundary(pp, pre_processed_data,  weight, b)
    return weight, b, acc, mse


def plot_decision_boundary(x_train, data, w, b):
    Y = data['species']
    x = x_train[:, 0]
    print(x_train)
    '''class1 = data.loc[Y == 1]
    class2 = data.loc[Y == -1]
    class1_1 = class1.drop('species', axis=1)
    class1_2 = class2.drop('species', axis=1)
    class1_1 = FeatureScalling(class1_1)
    class1_2 = FeatureScalling(class1_2)'''

    class1 = data.loc[Y == 1]
    y1 = class1['species']
    class1 = class1.drop('species', axis=1)

    class11 = FeatureScalling(class1)
    class11['species'] = y1
    class1 = class11

    class2 = data.loc[Y == -1]
    y2 = class2['species']
    class2 = class2.drop('species', axis=1)
    class22 = FeatureScalling(class2)
    class22['species'] = y2
    class2 = class22

    x_values = [np.min(x)-1, np.max(x)+1]
    print("xvalues")
    print(x_values)
    print("xvshape")
    print(np.shape(x_values))
    print("bw")
    print(w)
    print(np.shape(w))

    w = w.flatten()
    print("bw")
    print(w)
    print(np.shape(w))
    y_values = - (b + np.dot(w[0], x_values)) / w[1]
    y_values = y_values.flatten()
    print(y_values)
    '''yabs = abs(y_values)
    print("y")
    print(abs(y_values))
    print(yabs)
    yabss = np.array(yabs)
    print(yabss)'''

    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('class1')
    plt.ylabel('class2')
    plt.scatter(class1.iloc[:, 0], class1.iloc[:, 1], s=10, label='class1')
    plt.scatter(class2.iloc[:, 0], class2.iloc[:, 1], s=10, label='class2')
    plt.show()
