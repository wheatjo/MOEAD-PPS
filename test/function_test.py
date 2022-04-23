import numpy as np


def func(X):
    x_odd = X[:, 2::2]
    x_even = X[:, 1::2]
    sins = np.sin(0.5 * np.pi * X[:,0]).reshape(X.shape[0],1)
    coss =  np.cos(0.5 * np.pi * X[:,0]).reshape(X.shape[0], 1)
    g1 = np.sum(np.square(x_odd - sins), axis=1)
    g2 = np.sum(np.square(x_even - coss), axis=1)
    f1 = X[:, 0] + g1
    f2 = 1 - np.square(X[:, 0]) + g2
    return np.column_stack([f1, f2])


def evaluate(x):
    x_odd = x[:, 2::2]
    x_even = x[:, 1::2]
    g1 = np.sum(np.square(np.subtract(x_odd, np.sin(0.5*np.pi*x[:, 0]))), axis=1)
    g2 = np.sum(np.square(np.subtract(x_even, np.cos(0.5*np.pi*x[:, 0]))), axis=1)
    f1 = x[:, 0] + g1
    f2 = 1 - np.square(x[:, 0]) + g2
    c1 = (0.51 - g1) * (g1 - 0.5)
    c2 = (0.51 - g2) * (g2 - 0.5)
    return np.column_stack([f1, f2]), np.column_stack([c1, c2])


X = np.array([[1,2,3,4,5],[2,3,4,5,6]])
X = np.random.random((300,30))
res= func(X)
print(res)
a = np.sin(0.5 * np.pi * X[:, 0])