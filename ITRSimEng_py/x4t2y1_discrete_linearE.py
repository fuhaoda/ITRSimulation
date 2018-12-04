# -*- coding: utf-8 -*-
"""
This is an implecation of the simulation setting with the complex subgroups of X described in:
Su, X., C.-L. Tsai, H. Wang, D. M. Nickerson, and B. Li (2009). Subgroup analysis via
recursive partitioning. Journal of Machine Learning Research 10 (Feb), 141{158.
Modify the g(x) t(a) d(x,a) function accordingly

@author: yujiad2
"""

import sys
from ITRSimpy import *



TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "001_x4t2y1_discrete_linearE"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, redefine the generate function if needed
    """
    def generate(self, var_type, sample_size, dim=1, low=0, high=1):
        if var_type.lower() == 'cont':            
            return self.state.choice(np.arange(low,high+0.02,0.02), size=(sample_size, dim))
        elif var_type.lower() == 'ord' or var_type.lower() == 'nom':
            return self.state.randint(low, high+1, size=(sample_size, dim))
        else:
            return None

def x_func(sample_size, dg):
    """
    This func will return the X matrix
    :param sample_size:
    :param dg:
    :return:
    """

    # User need to define the dimension (dim) and boundary (low and high) of the random generated numbers
    # User can also change the title of the X matrix
    cont_array = dg.generate('cont', sample_size, dim=4, low=0.02, high=1)
    cont_title = [f"X_Cont{i}" for i in range(cont_array.shape[1])]
    ord_array = dg.generate('ord', sample_size, dim=0, low=0, high=3)
    ord_title = [f"X_Ord{i}" for i in range(ord_array.shape[1])]
    nom_array = dg.generate('nom', sample_size, dim=0, low=0, high=1)
    nom_title = [f"X_Nom{i}" for i in range(nom_array.shape[1])]

    x_array = np.column_stack([nom_array, ord_array, cont_array])
    x_title = nom_title + ord_title + cont_title

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
    p = 0.5 * np.ones((x.shape[0],1))
    a = np.random.binomial(n_act - 1, p) + 1
    return a.reshape(-1, 1)

def y_func(x, a, ydim, dg):
    """
    :param beta: intercept
    """
    
    z1 = (x[:,0] <= 0.5).reshape(-1,1)
    z2 = (x[:,1] <= 0.5).reshape(-1,1)
    
    y = 2 + 2*(a-1) + 2*z1 +2*z2 + 2*(a-1)*z1 + 2*(a-1)*z2 + dg.uniform(low=-np.sqrt(3),high=np.sqrt(3), size=[x.shape[0],1])
    return y.reshape(-1,1)


def main():
    """

    :return:
    """

    dg = CaseDataGenerator(seed=1)
    s = SimulationEngine(x_func=x_func,
                         a_func=a_func,
                         y_func=y_func,
                         training_size=TRAINING_SIZE,
                         testing_size=TESTING_SIZE,
                         n_act=NUMBER_RESPONSE,
                         ydim=Y_DIMENSION,
                         generator=dg)
    s.generate()
    s.export(OUTPUT_PREFIX)
        
if __name__ == "__main__":
    main()
