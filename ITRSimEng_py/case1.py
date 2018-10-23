from ITRSimEng_py.ITRSimpy import *

# need to define the a_func and y_func


def a_func(x, n_resp):
    beta_a = [-2.5, 3, 0, 1, 1]
    z = np.matmul(x, np.array(beta_a).reshape(-1, 1))
    p = 1 / (1 + np.exp(-z))
    a = np.random.binomial(n_resp - 1, p) + 1
    return a.reshape(-1, 1)


def y_func(x, a):
    beta_y = [[-2, 3, 0, 1, 1],
              [-2, 3, 0, 1, 1]]
    t_y = [[0.5],
           [0.5]]
    ydim = len(beta_y)
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
    s = SimulationEngine(a_func=a_func,
                         y_func=y_func,
                         training_size=100,
                         testing_size=1000,
                         n_cont=2,
                         n_ord=1,
                         n_nom=1,
                         n_resp=2,
                         ydim=2,
                         generator=g)
    s.generate()  # The naming seems confusing, data generator vs engine generate
    s.export("case1")
    test_ys = s.tys()
    test_azero = s.azero(test_ys)

    pd.DataFrame(np.insert(test_ys, 1, test_azero, axis=1),
                 columns=['A', 'A0', 'Y1_0', 'Y1_1', 'Y2_0', 'Y2_1']).astype({'A': 'int',
                                                                              'A0': 'int'}).to_csv("test_Ys.csv",
                                                                                                   index_label="ID")


if __name__ == "__main__":
    main()