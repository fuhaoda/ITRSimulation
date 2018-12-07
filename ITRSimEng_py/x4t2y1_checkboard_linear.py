
# coding: utf-8

# In[7]:

import sys
from ITRSimpy import *


# This is an checkboard of using ITR Simulation Engine
# Here we assume a two-dimensional 3*3 checkboard, 5 cells belongs to class 1 and 4 cells belongs to class 2.
# The current a_func is Random Bernoulli with p = 0.5 and the y_func is product of X and A
# Number of covariates is 4 and number of true covariates is 2

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_ACT = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "case_checkboard"


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
    ord_title = [f"X_Odd{i}" for i in range(ord_array.shape[1])]
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



def a_func(x, n_act):
    """
    Users define the function of A here. 
    A takes 1 and -1 with probability 0.5
    """
    p = np.ones((x.shape[0],1)) 
    a = 2*dg.binomial(n_act - 1, p) - 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim):
    """
    Users define the function of Y here. 
    y = A*(1{X1<1/3 or X1>2/3}*1{X2<1/3 or X2>2/3})
    
    """
    y = np.multiply((np.logical_or(x[:, [0]]<1/3, x[:, [0]]>2/3)*2-1)*(np.logical_or(x[:, [1]]<1/3, x[:, [1]]>2/3)*2-1),a)
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


if __name__ == "__main__":
    main()


# In[ ]:



