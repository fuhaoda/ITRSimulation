import sys
from ITRSimpy import *


# This is an example of using ITR Simulation Engine
# The current a_func is a linear model of X and the y_func is a linear model of X and A
# User can define customized a_func and y_func


def a_func(x, n_resp):
    """
    Users define the function of A here. The example below uses a linear function:
        A = -2.5 + 3*X0 + 0*X1 + 1*X2 + 1*X3 + 0*X4 + 0*X5

    :param x: the input X matrix for the a_function
    :param n_resp: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A
    """
    beta_a = [-2.5, 3, 0, 1, 1, 0, 0]
    z = np.matmul(x, np.array(beta_a).reshape(-1, 1))
    p = 1 / (1 + np.exp(-z))
    a = np.random.binomial(n_resp - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim):
    """
    Users define the function of Y here. The example below uses two linear functions to calculate
    the two dimension of Y:
        Y = A + (A-1.5)*[X2>0.7 && X4==0] + 2*X1 + rnorm
    :param x: the input X matrix for the a_function
    :param a: a n x 1 matrix of A
    :return:
    """

    y = a + \
        np.multiply((a - 1.5), np.logical_and(x[:, [1]] > 0.7, x[:, [3]] == 0)) + \
        np.multiply(2, x[:, [0]]) + \
        np.random.randn(x.shape[0], ydim)
    return y


# Modify section below accordingly
g = DataGenerator(seed=1)
s = SimulationEngine(a_func=a_func,
                     y_func=y_func,
                     training_size=500,
                     testing_size=50000,
                     n_cont=4,
                     n_ord=1,
                     n_nom=1,
                     n_resp=2,
                     ydim=1,
                     generator=g)
s.generate()
s.export(sys.argv[0].split(".")[0])
test_ys = s.tys()
testing_size = test_ys.shape[0]
test_azero = s.azero(test_ys)
test_ys_df = pd.DataFrame(test_ys.reshape(testing_size, -1),
                          columns=s.get_testcol())
test_ys_df['A'] = s.testing_data.act
test_ys_df['A0'] = test_azero
test_ys_df.to_csv("case1_test_Ys.csv", index_label="ID")
