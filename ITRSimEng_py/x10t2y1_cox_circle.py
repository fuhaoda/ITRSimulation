
# coding: utf-8

# In[4]:

import sys
from ITRSimpy import *


# Simulaiton setting for survival analysis through Cox model without considering censoring time
# The current a_func is Random Bernoulli with p = 0.5 
# Number of covariates is 10 and number of true covariates is 3

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_ACT = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "001_x10t2y1_cox_circle"

class DataGenerator(DataGenerator):
      
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
    cont_array = dg.generate('cont', sample_size, dim=10, low=0, high=1)
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


def a_func(x, n_act, dg):
    
    """
    Users define the function of A here. 
    A takes 1 and -1 with probability 0.5
    """
    p = 0.5*np.ones((x.shape[0],1)) 
    
    a = 2*dg.binomial(n_act - 1, p) - 1
    return a.reshape(-1, 1)



def y_func(x, a, ydim, dg):
    """
    Users define the function of Y here. 
    Y = 1-2X1 + X2 -X3 + 8(1-X1^2-X2^2)A
    

"""
    y_1 = - np.log(dg.uniform(0, 1, x.shape[0])).reshape(-1, 1)
    y_2 = 1 - np.multiply(2,x[:, [0]])+ x[:, [1]] - x[:, [2]] + 8*np.multiply(1-np.power(x[:, [0]],2)-np.power(x[:, [1]],2),a) +          dg.randn(x.shape[0], ydim)
        
    y = y_1/y_2
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



