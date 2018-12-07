
# coding: utf-8

# In[5]:

"""
@ author: Yubai Yuan   
"""

import sys
import math
from ITRSimpy import *

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 1


OUTPUT_PREFIX = "x2t0y1_spiral_linear"

class CaseDataGenerator(DataGenerator):
    """
    Define the Generator for each case, redefine the generate function if needed 
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
    cont_array = dg.generate('cont', sample_size, dim=2, low=-25, high=25)
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
  
    m = np.array(x).shape[0]
    p = 0.5*np.ones((m,1)) 
    a = 2*dg.binomial(n_act - 1, p) - 1
    return a.reshape(-1, 1)


def calculate_theta(x):
    
    ratio = x[1]/x[0]
    if math.atan(ratio)>0:
       theta_1 = math.atan(ratio) 
    else:
       theta_1 = math.atan(ratio) + np.pi
    if x[1]>0:
       theta = theta_1
    else:
       theta = theta_1+np.pi
    return(theta)

def calculate_radius(x):
    
    radius = x[0]**2 + x[1]**2

    return(radius)




def y_func(x, a, ydim, dg):

    x1 = pd.DataFrame(data=x)
    theta = x1.apply(calculate_theta,axis=1)
    radius = x1.apply(calculate_radius,axis=1)
    
    index_1 = np.array([],dtype = int)
    index_2 = np.array([],dtype = int)
    
    for i in range(0, 3): 
      a_1 = np.sqrt(radius.values) - 1.5*(theta.values+2*i*np.pi+0.5*np.pi)
      b_1 = np.where((a_1>0.1)*(a_1<2))[0]
      index_1 =   np.append(index_1,b_1)
    
    for i in range(0, 3): 
      a_2 = np.sqrt(radius.values) - 1.5*(theta.values+2*i*np.pi+1.5*np.pi)
      b_2 = np.where((a_2>0.1)*(a_2<2))[0]
      index_2 =   np.append(index_2,b_2)
    
    y = np.zeros((x1.shape[0],ydim))
    y[index_1] = 1
    y[index_2] = -1
    y = y*a
    
    return y

def main():
    """
    :return:
    """

    dg = DataGenerator(seed=1)
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


# In[ ]:



