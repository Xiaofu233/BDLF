'''
Auther: Bingyu_Hu
Mail: hby0728@mail.ustc.edu.cn
Date: 2023.12.29
'''

import numpy as np

#Create Variable Class
class Variable:
    def __init__(self, data):
        self.data = data


#Create Function Class
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == '__main__':
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    x.data = np.array(2.0)
    print(x.data)
    x = np.array(1.0)
    print(x.ndim)
    x = np.array([1,2,3])
    print(x.ndim)
    x = np.array([[1,2,3],[4,5,6]])
    print(x.ndim)
    x = Variable(np.array(2.0))
    f = Square()
    y = f(x)
    print(y.data)
    print(type(y))
    dy = numerical_diff(f, x)
    print(dy)
