# -*- coding: utf-8 -*-
"""
This is an implecation of the simulation setting of "Convergence in observational studies" of
"Estimating optimal treatment regimes via sbugroup identification in randomized contro trials and observational studies"
 
tunning parameter: b, i in a_func
                   theta in y_func

@author: yujiad2
"""

import sys
from ITRSimpy import *


# This is an example of using ITR Simulation Engine
# The current a_func is a linear model of X and the y_func is a linear model of X and A
# User can define customized a_func and y_func

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "001_w"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, redefine the generate function if needed (like constrained on a donut-shape space)
    """
    pass

def x_func(sample_size, dg):
    """
    This func will return the X matrix
    :param sample_size:
    :param dg:
    :return:
    """

    # User need to define the dimension (dim) and boundary (low and high) of the random generated numbers
    # User can also change the title of the X matrix
    cont_array = dg.generate('cont', sample_size, dim=5, low=0, high=1)
    cont_title = [f"X_Cont{i}" for i in range(cont_array.shape[1])]
    ord_array = dg.generate('ord', sample_size, dim=0, low=0, high=1)
    ord_title = [f"X_Ord{i}" for i in range(ord_array.shape[1])]
    nom_array = dg.generate('nom', sample_size, dim=0, low=0, high=1)
    nom_title = [f"X_Nom{i}" for i in range(nom_array.shape[1])]

    x_array = np.column_stack([cont_array, ord_array, nom_array])
    x_title = cont_title + ord_title + nom_title

    """
    # Or user can import a X matrix from the csv file:
    x_df = pd.read_csv("filename")
    x_array = np.asarray(x_df)
    x_title = list(x_df)
    """

    return x_title, x_array


def a_func(x, n_act, dg):
    """

    :param x: the input X matrix for the a_function
    :param n_act: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A
    """
    b = 6.5
    i = 1 # the related column of x in determining p
    z = -0.5 * b + b * x[:, i-1]
    p = 1 / (1 + np.exp(-z))
    a = dg.binomial(n_act - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim, dg):
    """
    Treatment a coded as 1/2
    """
    theta = 0.5
    y = 1 + 2 * x[:, 1].reshape(-1,1) + \
        theta * np.multiply((a-1), (x[:, 0] > 0.5).reshape(-1,1)) + \
        theta * np.multiply((2-a), (x[:, 0] <= 0.5).reshape(-1,1)) + \
        np.random.randn(x.shape[0], ydim)
    return y


def main():
    """

    :return:
    """

    g = DataGenerator(seed=1)
    s = SimulationEngine(x_func=x_func,
                         a_func=a_func,
                         y_func=y_func,
                         training_size=TRAINING_SIZE,
                         testing_size=TESTING_SIZE,
                         n_act=NUMBER_RESPONSE,
                         ydim=Y_DIMENSION,
                         generator=g)
    s.generate()
    s.export(OUTPUT_PREFIX)

if __name__ == "__main__":
    main()
