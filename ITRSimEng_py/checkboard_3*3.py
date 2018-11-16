
# coding: utf-8

# In[ ]:

"""
simulaiton setting of the checkboard example:
Here we assume a two-dimensional 3*3 checkboard, 5 cells belongs to class 1 and 4 cells belongs to class 2.

"""
def a_func(x, n_resp):
    """
    Users define the function of A here. 
    A takes 1 and -1 with probability 0.5
    """
    p = np.ones((x.shape[0],1)) 
    a = 2*np.random.binomial(n_resp - 1, p) - 1
    return a.reshape(-1, 1)


def y_func(x, a, ydim):
    """
    Users define the function of Y here. 
    y = A*(1{X1<1/3 or X1>2/3}*1{X2<1/3 or X2>2/3})
    
    """
    y = np.multiply((np.logical_or(x[:, [0]]<1/3, x[:, [0]]>2/3)*2-1)*(np.logical_or(x[:, [1]]<1/3, x[:, [1]]>2/3)*2-1),a)
    return y


# In[ ]:



