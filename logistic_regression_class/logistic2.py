"""
logistic2.py

calculate cross-entropy error function
"""

import numpy as np

N = 100
D = 2

# X is 100 2-dimensional points
X = np.random.randn(N, D)

# center the first 50 points at (-2, -2)
X[:50, :] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# first 50 labels are 0, last 50 labels are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialized weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(cross_entropy(T, Y))

# closed form solution
w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)
print(cross_entropy(T, Y))
