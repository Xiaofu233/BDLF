'''
Auther: Bingyu_Hu
Mail: hby0728@mail.ustc.edu.cn
Date: 2023.12.29
'''

import numpy as np

#Create Variable Class
class Variable:
    #用变量记录它的‘连接’
    def __init__(self, data):
        #仅支持nparray数据
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    #递归实现
    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    #循环实现
    def backward(self):
        #事先定义最终输出y.grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

    
#在numpy中，用0维的nparray计算，结果可能是np.float64,np.float32等标量，所以需要对output判断
# eg: x = np.array(1.0),x为0维，y = x ** 2, y的类型为np.float64
# eg: x = np.array([1.0]),x为1维，y = x ** 2, y为1维nparray
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
    

#Create Function Class
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        self.input = input
        output.set_creator(self)
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()





class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)


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

    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a 
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
