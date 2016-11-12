"""
logistic3.py

update weights using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50,:] - 2*np.ones((50,D))
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)

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

"""
shapes of things

Xb is (N,D+1)
w is (D+1,)
z, Y and T are (N,)
Xb.T is (D+1,N)
Xb.T.dot(T-Y) is (D+1,) same as w

hence the update rule works
"""

# do gradient descent 100 times
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    w += learning_rate * Xb.T.dot(T - Y)

    Y = sigmoid(Xb.dot(w))

print("Final w: ", w)

# plot the data and separating line
plt.scatter(X[:,0], X[:, 1], c=T, s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = w[0] + x_axis*(-w[2] / w[1])
plt.plot(x_axis, y_axis)
plt.show()
