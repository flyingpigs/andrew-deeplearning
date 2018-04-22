import numpy as np
import math


# 1 Building basic functions with numpy
# 1.1用numpy模块实现sigmoid函数

# sigmoid()函数
def basic_sigmoid(x):
    """
    Compute sigmoid of x
    :param x: a scalar
    :return:s: sigmoid(x)
    """
    s = 1.0 / (1 + math.exp(x))
    return s


# 使用numpy实现 sigmoid
def numpy_sigmoid(x):
    """
    Compute the sigmoid of x
    :param x: a scalar or numpy array of any size
    :return: s: sigmoid(x)
    """
    s = 1.0 / (1 + np.exp(x))
    return s


# 1.2 Sigmoid gradient

# 对s求导
def sigmoid_derivative(x):
    """
    Compute the gradient of the sigmoid function with respect to its input x
    :param x: a scalar or numpy array
    :return:ds: your computed gradient
    """
    s = 1.0 / (1 + np.exp(x))
    ds = s * (1 - s)
    return ds


# 1.3 Reshaping arrays
# x.shape is used to get the shape of a matrix/vector x
# x.reshape(...) is used to reshape x into some other dimension（dimension:按规格尺寸切割）

# 练习 实现image2vector() 将三维矩阵转化为一维
def image2vector(image):
    """
    :param image: a numpy array of the shape(length, height, depth)
    :return: v: a vector of shape(length * length * depth, 1)
    """
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2]), 1)
    return v


# 1.4 Normalizing rows(标准化)
# 标准化是指将x改为 x/||x||(x的每一行向量除以其范数)

# 练习 实现normalizeRows() 标准化一个矩阵的每一行
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x
    :param x: a numpy matrix of shape(n, m)
    :return: x: the normalized (by row) numpy matrix. you are allowed to modify x
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)    # x的范数 axis每一行求范数(以行为轴) keepdims保持二维特性
    x = x / x_norm    # numpy的广播效应
    return x


# 1.5 Broadcasting and the soft max function

# 练习 实现一个softmax函数 softmax(x) = exp(x) / sum(exp(x))
def softmax(x):
    """
    Implement the softmax function
    :param x: a numpy matrix of shape(n,m)
    :return: s: a numpy matrix equal to the softmax of x, of shape(n, m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


if __name__ == '__main__':
    # 1.1 numpy_sigmoid test
    x1 = np.array([1, 2, 3])
    print(numpy_sigmoid(x1))

    # 1.2 sigmoid_derivative test
    x2 = np.array([1, 2, 3])
    print(sigmoid_derivative(x2))

    # 1.3 image2vector test
    image = np.array([[[1, 2], [2, 3], [3, 4]],
                      [[4, 5], [5, 6], [6, 7]],
                      [[7, 8], [8, 9], [9, 10]]])
    print("image2vector(image) = " + str(image2vector(image)))

    # 1.4 normalizeRows test
    x = np.array([
        [0, 3, 4],
        [1, 6, 4]
    ])
    print("normalizeRows(x) = " + str(normalizeRows(x)))

    # 1.5 softmax test
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]
    ])
    print("softmax(x) = " + str(softmax(x)))





