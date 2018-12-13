# -*- coding: utf-8 -*-
"""
Zhu, X. and A. Qu (2016). Individualizing drug dosage with longitudinal data. Statistics
in medicine 35 (24), 4474-4488.

@author: yujiad2
"""

import sys
from ITRSimpy import *
from scipy.linalg import toeplitz



TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 3
Y_DIMENSION = 10
OUTPUT_PREFIX = "001_x11t3y10_longitudinal_log"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, redefine the generate function if needed
    """
    def choice(self, low=0, high=1, size=1):
        return self.state.choice(np.arange(low,high+1,1),size)
    def multivariate_normal(self, mean, cov, size):
        return self.state.multivariate_normal(mean, cov, size)
    
def x_func(sample_size, dg):
    """
    This func will return the X matrix
    :param sample_size:
    :param dg:
    :return:
    """

    # User need to define the dimension (dim) and boundary (low and high) of the random generated numbers
    # User can also change the title of the X matrix
    cont_array = dg.generate('cont', sample_size, dim=11, low=0.75, high=2.25)
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
    a = dg.choice(low=1,high=n_act,size=[x.shape[0], 1])
    return a



def y_func(x, a, ydim, dg):
    """
    :param beta: intercept
    """
    beta = 0.5
    d = 1.2
    dic = {1:100, 2:300, 3:600}
    rho = 0.8
    D = np.array([dic[a[i,0]] for i in range(a.shape[0])]).reshape([-1,1])
    b = dg.uniform(low=0.5,high=1.5,size=(x.shape[0],1)) * (dg.binomial(1,0.5,[x.shape[0],1])*2-1) 
    cov = toeplitz(np.append(1,np.repeat(rho,9)))
    epsilon = dg.multivariate_normal(np.zeros(ydim), cov, x.shape[0])
    ita = x[:,0:-1]*beta + np.repeat(x[:,-1].reshape([-1,1])*b + d*np.log(D),10,axis=1) + epsilon
    y = np.exp(ita)
        
    return y


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
