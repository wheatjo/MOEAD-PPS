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


class Parent(object):

    def __init__(self, name=None, num=None):
        self.name = name
        self.num = num
        print("use parent init")

    def __new__(cls):
        print("use new")
        return object.__new__(cls)

    def put_message(self):
        print(self.name)

class Child(Parent):

    def __init__(self, name=None, num=None):
        self.c = 0
        super().__init__(name, num)

    def __new__(cls, name, num):
        return object.__new__(cls)


# d = Parent('da', 10)
#
# print(d.name)
# print(d.num)


c = Child('s', 2)

print(c)
c.put_message()