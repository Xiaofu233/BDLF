import unittest
import numpy as np
from BDLF.core import Variable
from BDLF.utils import plot_dot_graph

# class SquareTest(unittest.TestCase):
#     def test_forward(self):
#         x = Variable(np.array(2.0))
#         y = square(x)
#         excepted = np.array(4.0)
#         self.assertEqual(y.data, excepted)

#     def test_backward(self):
#         x = Variable(np.random.rand(1))
#         y = square(x)
#         y.backward()
#         num_grad = numerical_diff(square, x)
#         flg = np.allclose(x.grad, num_grad) #判断值是否接近
#         self.assertTrue(flg)

# unittest.main()



# def goldstein(x, y):
#     z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
#         (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
#     return z


# x = Variable(np.array(1.0))
# y = Variable(np.array(1.0))
# z = goldstein(x, y)
# z.backward()

# x.name = 'x'
# y.name = 'y'
# z.name = 'z'
# plot_dot_graph(z, verbose=False, to_file='goldstein.png')

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)
gx = x.grad
x.clear_grad()
gx.backward()
print(x.grad)
