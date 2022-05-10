import os.path
import sys

import numpy as np
from tempfile import TemporaryFile

# outfile = TemporaryFile()
#
a = np.random.random((3,10))
# path = sys.path[0]
#
# path = os.path.join(path, "data/a.npz")
# # print(path)
# # np.savez(path, a=a)
#
# o = np.load(path)
# print(o['a'])

def save_data(file_name, pop, arch):
    path = sys.path[0]
    path = os.path.join(path, "data")
    path = path + "/" + file_name + ".npz"
    np.savez(path, pop=pop, arch=arch)


if __name__ == '__main__':
    pass

