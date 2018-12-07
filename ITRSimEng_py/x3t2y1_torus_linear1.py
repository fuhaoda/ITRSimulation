# -*- coding: utf-8 -*-
"""
This is an implecation of the simulation setting with the boundary subgroups of X being a 3-D donut shape (a torus)
Modify the g(x) t(a) d(x,a) function accordingly

@author: yujiad2
"""

import sys
from ITRSimpy import *


# This is an example of using ITR Simulation Engine
# The current a_func is a linear model of X and the y_func is a linear model of X and A
# User can define customized a_func and y_func

TRAINING_SIZE = 500
TESTING_SIZE = 50000
NUMBER_RESPONSE = 2
Y_DIMENSION = 1
OUTPUT_PREFIX = "001_x3t2y1_torus_linear1"

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
    cont_array = dg.generate('cont', sample_size, dim=3, low=-1, high=1)
    cont_title = [f"X_Cont{i}" for i in range(cont_array.shape[1])]
    ord_array = dg.generate('ord', sample_size, dim=0, low=0, high=1)
    ord_title = [f"X_Ord{i}" for i in range(ord_array.shape[1])]
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
    randomly assigned with p=0.5

    :param x: the input X matrix for the a_function
    :param n_act: the number of possible responses, i.e. treatment options
    :return: a n x 1 matrix of A
    """
    p = 0.5 * np.ones((x.shape[0],1))
    a = dg.binomial(n_act - 1, p) + 1
    return a.reshape(-1, 1)

def g(x):
    """
    g(x) = 3*x_3
    """
    g = 3 * x[:,2].reshape(-1,1)
    return g

def t(a):
    """
    g(a) = 0.3*1{a==1} + 0.2*1{a==2}
    """
    t = 0.3 * (a == 1).reshape(-1,1) + \
        0.2 * (a == 2).reshape(-1,1)
    return t

def d(x, a):
    """
    torus shape
    :param R: the major radius of the torus
    :parm r: the minor radius of the torus
    :param theta: coefficient
    """
    try:
        assert x.shape[1] >=3
    except AssertionError:
        raise('The dimension of x is smaller than 3.')
    theta = 0.8
    R = 0.7
    r = 0.3
    cond = ((np.sqrt(x[:,0]**2 + x[:,1]**2) - R)**2 + x[:,2]**2).reshape(-1,1) 
    d = theta * np.multiply((a-1), cond <= r**2) + \
        theta * np.multiply((2-a), cond > r**2)
    return d

def y_func(x, a, ydim, dg):
    """
    :param beta: intercept
    """
    beta = 1
    y = beta + g(x) + t(a) + d(x,a) + np.random.randn(x.shape[0], ydim)
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
    illustration(s.training_data.x)
    s.export(OUTPUT_PREFIX)

def illustration(x_data):
    """
    illustration of the torus
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    n, R, r = 100, 0.7, 0.3
    theta = np.linspace(0, 2.*np.pi, n)
    phi = np.linspace(0, 2.*np.pi, n)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    cond = (((np.sqrt(x_data[:,0]**2 + x_data[:,1]**2) - R)**2 + x_data[:,2]**2)>r**2)
    fig = plt.figure()
    
    ax1 = fig.add_subplot(111, projection='3d')    
    colormap = {1:'r', 0:'b'}
    ax1.scatter(x_data[:,0], x_data[:,1], x_data[:,2], c=cond, cmap=plt.cm.Spectral)
    ax1.set_zlim(-1,1)
    #ax1.plot_wireframe(x,y,z,rstride=5, cstride=5)
    ax1.plot_surface(x, y, z, rstride=5, cstride=5, color='r', alpha=0.2)
    ax1.view_init(36, 36)
    plt.savefig(f'{OUTPUT_PREFIX}.png')
        
if __name__ == "__main__":
    main()
