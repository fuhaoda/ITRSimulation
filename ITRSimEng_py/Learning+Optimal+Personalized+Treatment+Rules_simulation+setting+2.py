
# coding: utf-8

# In[ ]:

"""
simulaiton setting 2 in the reference 
"Learning Optimal Personalized Treatment Rules in Consideration of Benefit and Risk: 
 with an Application to Treating Type 2 Diabetes Patients with Insulin Therapies"

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
    Y = 1-2X1 + X2 -X3 + 8(1-X1^2-X2^2)A
    
    """
    y = 1 - np.multiply(2,x[:, [0]])+ x[:, [1]] - x[:, [2]] + 8*np.multiply(1-np.power(x[:, [0]],2)-np.power(x[:, [1]],2),a) +         np.random.randn(x.shape[0], ydim)
    return y

