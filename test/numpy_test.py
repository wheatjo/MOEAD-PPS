import numpy as np

a = np.array([np.repeat(1,10)])
b = np.array([np.repeat(2,10)])
c = np.array([np.repeat(3,10)])

d = np.hstack((a.T,b.T,c.T))

print(d)

np.where()