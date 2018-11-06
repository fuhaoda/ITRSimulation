import sys
from ITRSimpy import *


# This is an example of using ITR Simulation Engine
# The current a_func is a linear model of X and the y_func is a linear model of X and A
# User can define customized a_func and y_func

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 2
OUTPUT_PREFIX = "case1"


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
    ord_title = [f"X_Cont{i}" for i in range(ord_array.shape[1])]
    nom_array = dg.generate('nom', sample_size, dim=1, low=0, high=1)
    nom_title = [f"X_Cont{i}" for i in range(nom_array.shape[1])]

    x_array = np.column_stack([cont_array, ord_array, nom_array])
    x_title = cont_title + ord_title + nom_title

    """
    # Or user can import a X matrix from the csv file:
    x_df = pd.read_csv("filename")
    x_array = np.asarray(x_df)
    x_title = list(x_df)
    """

    return x_title, x_array


def a_func(x, n_resp):
    """
    Users define the function of A here. The example below uses a linear function:
        P(A) = sigmoid( -2.5 + 3*X0 + 0*X1 + 1*X2 + 1*X3 + 0*X4 + 0*X5)

    :param x: the input X matrix for the a_function
    :param n_resp: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A
    """

    beta_a = [-2.5, 3, 0, 1, 1, 0, 0]
    assert len(beta_a) == x.shape[1]
    z = np.matmul(x, np.array(beta_a).reshape(-1, 1))
    p = 1 / (1 + np.exp(-z))
    a = np.random.binomial(n_resp - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim):
    """
    Users define the function of Y here. The example below uses two linear functions to calculate
    the two dimension of Y:
        Y_0 = -2 + 3*X0 + X2 + X5 + 0.5*A      (0 is omitted here)
        Y_1 = -2 + 4*X0 - X2 + X5 + 0.7*A
    :param x: the input X matrix for the a_function
    :param a: a n x 1 matrix of A
    :return:
    """
    beta_y = [[-2, 3, 0, 1, 0, 0, 1],
              [-2, 4, 0, -1, 0, 0, 1]]
    t_y = [[0.5],
           [0.7]]

    beta_t = np.append(beta_y, t_y, axis=1)
    assert beta_t.shape == (ydim, x.shape[1] + 1)

    y = np.matmul(np.append(x, a, axis=1), beta_t.T) + \
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
                         n_resp=NUMBER_RESPONSE,
                         ydim=Y_DIMENSION,
                         generator=g)
    s.generate()
    s.export(OUTPUT_PREFIX)
    test_ys = s.tys()
    testing_size = test_ys.shape[0]
    test_azero = s.azero(test_ys)
    test_ys_df = pd.DataFrame(test_ys.reshape(testing_size, -1),
                              columns=s.get_testcol())
    test_ys_df['A'] = s.testing_data.act
    test_ys_df['A0'] = test_azero
    test_ys_df.to_csv(f"{OUTPUT_PREFIX}_test_Ys.csv", index_label="ID")


if __name__ == "__main__":
    main()
