'''
Auther: Bingyu_Hu
Mail: hby0728@mail.ustc.edu.cn
Date: 2024.01.03
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
    
    def clear_grad(self):
        self.grad = None

    def backward(self):
        #事先定义最终输出y.grad
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx

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
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        self.inputs = inputs
        self.outputs = outputs
        for output in outputs:
            output.set_creator(self)
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)



class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
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
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)
    