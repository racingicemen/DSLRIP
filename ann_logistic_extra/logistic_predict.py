import numpy as np
from process import get_binary_data

X, Y = get_binary_data()

# randomly initialize weights
D = X.shape[1]
W = np.random.randn(D)

# bias term
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# given the inputs, weights and bias, calculate the output
# if x is an numpy array, x + 2 adds 2 to each element of x
def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
# output < 0.5 => class 0
# output > 0.5 => class 1
predictions = np.round(P_Y_given_X)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score: ", classification_rate(Y, predictions))
