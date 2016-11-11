"""
logistic_regression_class/logistic1.py
"""

import numpy as np

# number of samples
N = 100

# number of dimensions
D = 2

# X is a collection of N D-dimensional values, drawn from the standard normal distribution
# with mean = 0 and standard deviation = 1
# the ith row of X contains (X_i1, X_i2)
# shape of X is N x D or (N,D)
X = np.random.randn(N, D)

# ones will be a N x 1 column vector of ones
# shape of ones is N x 1 or (N,1)
ones = np.ones((N, 1))
# the above stament is equivalent to the following
# ones = np.array([[1]*N]).T
#   [1]*N creates a row of N 1s
#   np.array([...]) converts that into a numpy array
#   .T takes the transpose to get a column of N ones

# the ith row Xb contains (1, X_i1, X_i2) axis=1 indicates adding an extra column
# shape of Xb is N x (D+1) or (N,(D+1))
Xb = np.concatenate((ones, X), axis=1)

# weight matrix, initialized to normally distributed random values. We include the bias term,
# which is why the argument is D + 1
# shape of w is (D+1,) which is interpreted as a (D+1) x 1 or a 1 x (D+1) vector, whichever works
w = np.random.randn(D + 1)

# N x (D+1) dot (D+1) x 1 gives the shape of z as N x 1 or (N,)
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# shape of sigmoid(z) is (N,)
print(sigmoid(z))
