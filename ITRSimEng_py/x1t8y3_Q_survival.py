# -*- coding: utf-8 -*-
"""
Implement a simplified version of the simulation setting described in Goldberg, Y., & Kosorok, M. R. (2012). Q-learning with censored data. Annals of statistics, 40(1), 529.
Regularized outcome weighted subgroup identification for differential treatment effects. Biometrics, 71(3), 645-653.
Modify the g(x) t(a) d(x,a) function accordingly

@author: yujiad2
"""

import sys
from ITRSimpy import *



TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 8
Y_DIMENSION = 3
OUTPUT_PREFIX = "001_x1t8y3_Q_survival"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, redefine the generate function if needed
    """
    def choice(self, low=0, high=1, size=1):
        return self.state.choice(np.arange(low,high+1,1),size)

def x_func(sample_size, dg):
    """
    This func will return the X matrix
    :param sample_size:
    :param dg:
    :return:
    """

    # User need to define the dimension (dim) and boundary (low and high) of the random generated numbers
    # User can also change the title of the X matrix
    cont_array = dg.generate('cont', sample_size, dim=1, low=0.5, high=1)
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

def evolve(W0, T0, trt, dg): #trt:1=A, 0=B
    if trt == 1:
        W0_p = W0-0.5
        T0_p = T0/(8*W0)
    elif trt == 0: 
        W0_p = W0-0.25 
        T0_p = T0/(6*W0)
    
    W1 = np.max((W0_p+(1-W0_p)*(1-2**(-1/2))+ dg.randn(1,1)*0.1, 0.25))
    T1 = np.min((T0_p+4*T0_p/3+ dg.randn(1,1)*0.1, 1))
    S0 = 3*(W0_p+2)/(20*T0_p)
    return W1, T1, S0

def y_func(x, a_, ydim, dg):
    """
    :param beta: intercept
    """
    dic = {1:np.array([1,1,1]), 2:np.array([1,1,0]), 3:np.array([1,0,1]), 4:np.array([0,1,1]), \
           5:np.array([1,0,0]), 6:np.array([0,0,1]), 7:np.array([0,1,0]), 8:np.array([0,0,0])}
    y = np.zeros([x.shape[0], 3])
    
    for n in range(x.shape[0]): 
        a = dic[int(a_[n])]
        for k in range(3):
            if k==0:
                W0 = x[n]; T0 = 1;
            W0, T0, S0 = evolve(W0, T0, a[k], dg)
            y[n,k] = S0
        
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
