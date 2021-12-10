import numpy as np

# python 参数传递是浅拷贝
def simple(x):
    x = 0
    #print(x)
    return x

# c = np.random.random((2,4))
# print(c)
# simple(c)
#
# print(c)
c = {"k": simple}


a = c['k']

print(a(3))
