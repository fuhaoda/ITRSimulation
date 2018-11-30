"""
This is an implecation of the simulation setting of "improve numerical stability" of
"Estimating optimal treatment regimes via sbugroup identification in randomized contro trials and observational studies"

"""
import sys
from ITRSimpy import *

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "001_x4t2y1_dichotomous_linear1"

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
    cont_array = dg.generate('cont', sample_size, dim=4, low=0, high=1)
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
    randomly assigned with p=0.5

    :param x: the input X matrix for the a_function
    :param n_act: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A
    """
    p = 0.5*np.ones((x.shape[0],1))
    a = np.random.binomial(n_act - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim, dg):
    """
    Y = beta_0 + sign(X_2-0.5) + A*1{X_1<=0.6} +(1-A)*1{X_1>0.6}
    :param x: the input X matrix for the a_function
    :param a: a n x 1 matrix of A
    :return:
    """
    beta_0 = 5*np.ones((x.shape[0],1))
    y = beta_0 + np.sign(x[:,1] - 0.5).reshape(-1,1) + \
        np.multiply((a-1), (x[:,0] <= 0.6).reshape(-1,1)) + \
        np.multiply((2-a), (x[:,0] > 0.6).reshape(-1,1))
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
