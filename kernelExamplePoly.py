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
    KIy = polyKernelTrain(XTrain, yTrain, polyDegree, alpha=1.0)
    yPrediction = polyKernelTest(XTest, XTrain, KIy, polyDegree)
    mse = meanSquareError(yPrediction, yTest)
    rms = rootMeanSquareError(yPrediction, yTest, N)
    print("meanSquareError, polynomial kernel (deg = ", s, ") = ", mse, sep='')
    print("rootMeanSquareError, polynomial kernel (deg = ", s, ") = ", rms, sep='')
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


# Polynomial fit
ss = [1, 2, 3, 5, 10]
results = []
rmsBest = 2 ** 50
sBest = None
for s in ss:
    yPredict, mse, rms = polyKernel(XTrain, yTrain, XTest, yTarget, s, alpha=1.0)
    results.append(rms)
    if rms < rmsBest:
        sBest = s
        rmsBest = rms
        yTest = yPredict
print("degreeBest =", sBest)
print("rmsBest =", rmsBest)

# Plots
plt.scatter(xTrain, yTrain)
plt.plot(xTest, yTarget, color='lime')
plt.plot(xTest, yTest, color='r')
plt.title('Polynomial, degree = 3')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
