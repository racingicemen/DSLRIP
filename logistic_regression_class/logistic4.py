import numpy as np

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

# do gradient descent 100 times
learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    # gradient descent weight update with regularization
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w)

    Y = sigmoid(Xb.dot(w))

print("Final w: ", w)
