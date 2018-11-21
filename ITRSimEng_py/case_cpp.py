import sys
from ITRSimpy import *


# This is an example of using ITR Simulation Engine
# The current a_func is a linear model of X and the y_func is a linear model of X and A
# User can define customized a_func and y_func

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_ACT = 2
Y_DIMENSION = 2
OUTPUT_PREFIX = "case_cpp"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, keep the same seed for all the random data generating procedure.
    Add more functions if needed in generating random numbers
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
    cont_array = dg.generate('cont', sample_size, dim=4, low=0, high=1)
    cont_title = [f"X_Cont{i}" for i in range(cont_array.shape[1])]
    ord_array = dg.generate('ord', sample_size, dim=2, low=0, high=1)
    ord_title = [f"X_Odd{i}" for i in range(ord_array.shape[1])]
    nom_array = dg.generate('nom', sample_size, dim=1, low=0, high=1)
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
    Users define the function of A here. The example below uses a linear function:
        A = -2.5 + 3*X0 + 0*X1 + 1*X2 + 1*X3 + 0*X4 + 0*X5

    :param x: the input X matrix for the a_function
    :param n_act: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A (starging from 1)
    """
    beta_a = [-2.5, 3, 0, 1, 1, 0, 0]
    z = np.matmul(x, np.array(beta_a).reshape(-1, 1))
    p = 1 / (1 + np.exp(-z))
    a = dg.binomial(n_act - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim, dg):
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
        dg.randn(x.shape[0], ydim)
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
                         n_act=NUMBER_ACT,
                         ydim=Y_DIMENSION,
                         generator=g)
    s.generate()
    s.export(OUTPUT_PREFIX)
    #s.print_training_data(15)
    #s.print_testing_data(15)



if __name__ == "__main__":
    main()
