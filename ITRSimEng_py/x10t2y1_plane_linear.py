

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
OUTPUT_PREFIX = "case3_1"


class CaseDataGenerator(DataGenerator):
  
      pass


def a_func(x, n_resp):
    """
    Users define the function of A here. 
    A takes 1 and -1 with probability 0.5
    """
    p = np.ones((x.shape[0],1)) 
    a = 2*dg.binomial(n_resp - 1, p) - 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim):
    """
    Users define the function of Y here. 
    Y = 1-2X1 + X2 -X3 + 2(1-X1-X2)A
    
    """
    y = 1 - np.multiply(2,x[:, [0]])+ x[:, [1]] - x[:, [2]] + 2*np.multiply(1-x[:, [0]]-x[:, [1]],a) + \               
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
                         n_act=NUMBER_ACT,
                         ydim=Y_DIMENSION,
                         generator=g)
    s.generate()
    s.export(OUTPUT_PREFIX)


if __name__ == "__main__":
    main()



