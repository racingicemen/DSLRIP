import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

twos = 2*np.ones((50, D))

X[:50,:] = X[:50, :] - twos
X[50:,:] = X[50:, :] + twos

T = np.array([0]*50 + [1]*50)

ones = np.ones((N,1))
Xb = np.concatenate((ones, X), axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.array([0, 4, 4])

z = Xb.dot(w)
Y = sigmoid(z)

# c=color c=T works because first 50 are in class 0, next 50 in class 1
# s=size of dot
# alpha = transparency of each dot
plt.scatter(X[:,0], X[:, 1], c=T, s=100, alpha=0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis) # plotting the straight line y=-x
plt.show()
