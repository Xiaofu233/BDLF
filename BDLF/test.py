import unittest
from stage1 import Variable, Function, square, numerical_diff
import numpy as np

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)

    def test_backward(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) #判断值是否接近
        self.assertTrue(flg)

unittest.main()
