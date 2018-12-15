
# coding: utf-8

# In[4]:

import sys
from ITRSimpy import *


# Simulaiton setting 1 in the reference 
# "Learning Optimal Personalized Treatment Rules in Consideration of Benefit and Risk: 
# With an Application to Treating Type 2 Diabetes Patients with Insulin Therapies"
# The current a_func is Random Bernoulli with p = 0.5 
# Number of covariates is 10 and number of true covariates is 3

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_ACT = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "x10t3y1_linear_plane"


class CaseDataGenerator(DataGenerator):
  
      def choice(self, low=0, high=1, size=1):
        return self.state.choice(np.arange(1,4,1),size)
  

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
    m = np.array(x).shape[0]
    a = dg.choice(low=1,high=4,size=[m, 1])
    return a.reshape(-1, 1)



def y_func(x, a, ydim,dg):
    """
    Users define the function of Y here. 
    Y = 1-2X1 + X2 -X3 + I(X1+X2+X3<1.5)I(A=1)+I(1.5=<X1+X2+X3<2.5)I(A=2)+I(X1+X2+X3>=2.5)I(A=3)
    
    """
    m = np.array(x).shape[0]
    y = 1 - np.multiply(2,x[:, [0]])+ x[:, [1]] - x[:, [2]] +         np.multiply(np.logical_or(x[:, [0]]+ x[:, [1]] + x[:, [2]]<1.5,x[:, [0]]+ x[:, [1]] + x[:, [2]]>0), a==1)+        np.multiply(np.logical_or(x[:, [0]]+ x[:, [1]] + x[:, [2]]<2.5,x[:, [0]]+ x[:, [1]] + x[:, [2]]>=1.5), a==2)+        np.multiply(np.logical_or(x[:, [0]]+ x[:, [1]] + x[:, [2]]<3,x[:, [0]]+ x[:, [1]] + x[:, [2]]>=2.5), a==3)+        + dg.randn(m, ydim)
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
                         n_act=NUMBER_ACT,
                         ydim=Y_DIMENSION,
                         generator=dg)
    s.generate()
    s.export(OUTPUT_PREFIX)


if __name__ == "__main__":
    main()


# In[ ]:



