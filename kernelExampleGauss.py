import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import scipy.spatial as spt
import scipy.sparse.linalg as sla

import matplotlib.pyplot as plt

rnd.seed(8675309)

def create_data_2D(n, w, xmin=-4, xmax=8):
    x = rnd.random(n) * (xmax - xmin) + xmin
    # Vandermonde matrix, V[i][j] = v[i] ** j
    X = np.vander(x, len(w))
    y = np.dot(X, w) + rnd.randn(n) * 0.5
    return x, y

def meanSquareError(prediction, target):
    return 0.5 * np.sum( (prediction - target) ** 2 )

def rootMeanSquareError(prediction, target, N):
    error = meanSquareError(prediction, target)
    return ( 2 * error / N ) ** 0.5


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

def gaussKernelTrain(XTrain, yTrain, s=1):
    K = gaussKernelMatrix(XTrain, s)
    KI = la.inv(K + 1. * np.identity(n))
    KIy = np.dot(KI, yTrain)
    return KIy

def gaussKernelTest(XTest, XTrain, KIy, s=1):
    yTest = np.zeros(N)
    for i in range(N):
        k = gaussKernelVector(XTest[i,:], XTrain, s)
        yTest[i] = np.dot(k, KIy)
    return yTest

def gaussKernel(XTrain, yTrain, XTest, yTest, s=1):
    n = yTrain.size
    N = yTest.size
    KIy = gaussKernelTrain(XTrain, yTrain, s)
    yPrediction = gaussKernelTest(XTest, XTrain, KIy, s)
    mse = meanSquareError(yPrediction, yTest)
    rms = rootMeanSquareError(yPrediction, yTest, N)
    print("meanSquareError, gaussian kernel (s = ", s, ") = ", mse, sep='')
    print("rootMeanSquareError, gaussian kernel (s = ", s, ") = ", rms, sep='')
    return yPrediction, mse, rms


# Create training data
n = 50
w = np.array([0.1, -0.8, 0.0, 11.5])
xTrain, yTrain = create_data_2D(n, w)
XTrain = np.vander(xTrain, 2)

# Create test data and target variables
N = 2 * n
xTest = np.linspace(-4, 8, N)
XTest = np.vander(xTest, 2)

XTarget = np.vander(xTest, len(w))
yTarget = np.dot(XTarget, w)


# Gaussian fit
ss = [0.01, 0.1, 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2, 3, 5, 10]
results = []
rmsBest = 2 ** 50
sBest = None
for s in ss:
    yPredict, mse, rms = gaussKernel(XTrain, yTrain, XTest, yTarget, s)
    results.append(rms)
    if rms < rmsBest:
        sBest = s
        rmsBest = rms
        yTest = yPredict
print("varianceBest =", sBest)
print("rmsBest =", rmsBest)

# Plots
plt.scatter(xTrain, yTrain)
plt.plot(xTest, yTarget, color='lime')
plt.plot(xTest, yTest, color='r')
plt.title('Gaussian, variance = 1.5')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
