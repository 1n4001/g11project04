import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import scipy.spatial as spt
import scipy.sparse.linalg as sla

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def meanSquareError(prediction, target):
    return 0.5 * np.sum( (prediction - target) ** 2 )

def rootMeanSquareError(prediction, target, N):
    error = meanSquareError(prediction, target)
    return ( 2 * error / N ) ** 0.5

# Polynomial Kernel

def polyKernelMatrix(X, polyDegree):
    return ( np.dot(X, X.T) + 1.0 ) ** polyDegree

def polyKernelVector(x, X, polyDegree):
    return ( np.dot(x, X.T) + 1.0 ) ** polyDegree

def polyKernelTrain(XTrain, yTrain, polyDegree=1, alpha=1.0):
    n = yTrain.size
    K = polyKernelMatrix(XTrain, polyDegree)
    KI = la.inv( K + alpha * np.identity(n) )
    KIy = np.dot( KI, yTrain)
    return KIy

def polyKernelTest(XTest, XTrain, KIy, polyDegree=1):
    N = XTest[:,0].size
    yTest = np.zeros(N)
    for i in range(N):
        k = polyKernelVector(XTest[i,:], XTrain, polyDegree)
        yTest[i] = np.dot(k, KIy)
    return yTest

def polyKernel(XTrain, yTrain, XTest, yTest, polyDegree=1, alpha=1.0):
    n = yTrain.size
    N = yTest.size
    KIy = polyKernelTrain(XTrain, yTrain, polyDegree, alpha)
    yPrediction = polyKernelTest(XTest, XTrain, KIy, polyDegree)
    #mse = meanSquareError(yPrediction, yTest)
    #rms = rootMeanSquareError(yPrediction, yTest, N)
    #print("meanSquareError, polynomial kernel (deg = ", polyDegree, ") = ", mse, sep='')
    #print("rootMeanSquareError, polynomial kernel (deg = ", polyDegree, ") = ", rms, sep='')
    return yPrediction, KIy#, mse, rms

# Gaussian Kernel

def squaredEDM(X):
    V = spt.distance.pdist(X, 'sqeuclidean')
    D = spt.distance.squareform(V)
    return D

def gaussKernelMatrix(X, s):
    D = squaredEDM(X)
    K = np.exp( -0.5 / s ** 2 * D )
    return K

def gaussKernelVector(x, X, s):
    d = np.sum( (X - x) ** 2.0, axis=1 )
    k = np.exp( -0.5 / s ** 2 * d )
    return k

def gaussKernelTrain(XTrain, yTrain, s=1, alpha=1.0):
    n = yTrain.size
    K = gaussKernelMatrix(XTrain, s)
    KI = la.inv(K + alpha * np.identity(n))
    KIy = np.dot(KI, yTrain)
    return KIy

def gaussKernelTest(XTest, XTrain, KIy, s=1):
    N = XTest[:,0].size
    yTest = np.zeros(N)
    for i in range(N):
        k = gaussKernelVector(XTest[i,:], XTrain, s)
        yTest[i] = np.dot(k, KIy)
    return yTest

def gaussKernel(XTrain, yTrain, XTest, yTest, s=1, alpha=1.0):
    n = yTrain.size
    N = yTest.size
    KIy = gaussKernelTrain(XTrain, yTrain, s, alpha)
    yPrediction = gaussKernelTest(XTest, XTrain, KIy, s)
    #mse = meanSquareError(yPrediction, yTest)
    #rms = rootMeanSquareError(yPrediction, yTest, N)
    #print("meanSquareError, gaussian kernel (s = ", s, ") = ", mse, sep='')
    #print("rootMeanSquareError, gaussian kernel (s = ", s, ") = ", rms, sep='')
    return yPrediction, KIy#, mse, rms

def insertOnes(x):
    if x.ndim == 1:
        X = np.vander(x, 2, increasing=True)
    else:
        X = np.insert(x, 0, 1.0, axis=1)
    return X


df = pd.read_csv('Data\data_randomized.csv')

X = df.iloc[:, [0,1,2,3]].values
X[:, [0]] = X[:, [0]] / 365
X = insertOnes(X)
y = df.iloc[:, 5].values
#print(X)
#print(y)

X_train, X_finalTest, y_train, y_finalTest = train_test_split(X, y, test_size=0.2, random_state=8675309)

# Parameters to test
alphas = [0.001, 0.005, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012,
    0.0125, 0.0127, 0.0129, 0.013, 0.0131, 0.0133, 0.0135,
    0.014, 0.015, 0.1, 0.15, 0.2, 0.21, 0.22, 0.23, 0.24,
    0.25, 0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8, 1.0]
degrees = [1,2,3,4,5,6,9,15]
gammas = [0.01, 0.1, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
    0.26, 0.265, 0.27, 0.275, 0.28, 0.29, 0.3, 0.31, 0.33, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.7, 1.0, 2.0, 2.5, 2.7, 2.9, 3.0, 3.1, 3.3, 3.5, 4.0, 5.0, 10.0]

kf = KFold(n_splits=5, random_state=8675309)
print("kf =", kf)

results = []
rmsBest = 2 ** 50
fold = 0
for train_index, test_index in kf.split(X_train):
    #print("Train:", train_index, "\nTest:", test_index)
    XTrain, XTest = X_train[train_index], X_train[test_index]
    yTrain, yTest = y_train[train_index], y_train[test_index]
    #print("XTrain", XTrain, "\nyTrain", yTrain)
    N = yTest.size
    fold += 1

    for alpha in alphas:
        for deg in degrees:
            #deg = 2
            #alpha = 1.0
            yPredictPoly, KIy = polyKernel(XTrain, yTrain, XTest, yTest, polyDegree=deg, alpha=alpha)
            msePoly = meanSquareError(yPredictPoly, yTest)
            rmsPoly = rootMeanSquareError(yPredictPoly, yTest, N)
            results.append( (fold, "poly", deg, alpha, msePoly, rmsPoly) )
            if rmsPoly < rmsBest:
                rmsBest = rmsPoly
                yBest = yPredictPoly
                KIyBest = KIy
                XTrainBest = XTrain
                resultBest = (fold, "poly", deg, alpha, msePoly, rmsPoly)

    for alpha in alphas:
        for gamma in gammas:
            #gamma = 1
            #alpha = 1.0
            yPredictGauss, KIy = gaussKernel(XTrain, yTrain, XTest, yTest, s=gamma, alpha=alpha)
            mseGauss = meanSquareError(yPredictGauss, yTest)
            rmsGauss = rootMeanSquareError(yPredictGauss, yTest, N)
            results.append( (fold, "gauss", gamma, alpha, mseGauss, rmsGauss) )
            if rmsGauss < rmsBest:
                rmsBest = rmsGauss
                yBest = yPredictGauss
                KIyBest = KIy
                XTrainBest = XTrain
                resultBest = (fold, "gauss", gamma, alpha, mseGauss, rmsGauss)

#print("results =", results)
print("Best result:", resultBest)

if resultBest[1] == "poly":
    N = y_finalTest.size
    yFinalPredict = polyKernelTest(X_finalTest, XTrainBest, KIyBest, resultBest[2])
    mseFinal = meanSquareError(yFinalPredict, y_finalTest)
    rmsFinal = rootMeanSquareError(yFinalPredict, y_finalTest, N)
    print("meanSquareErrorFinal =", mseFinal)
    print("rootMeanSquareErrorFinal =", rmsFinal)
elif resultBest[1] == "gauss":
    N = y_finalTest.size
    yFinalPredict = gaussKernelTest(X_finalTest, XTrainBest, KIyBest, resultBest[2])
    mseFinal = meanSquareError(yFinalPredict, y_finalTest)
    rmsFinal = rootMeanSquareError(yFinalPredict, y_finalTest, N)
    print("meanSquareErrorFinal =", mseFinal)
    print("rootMeanSquareErrorFinal =", rmsFinal)
else:
    print("Error: Best result was neither \"poly\" nor \"gauss\"")
