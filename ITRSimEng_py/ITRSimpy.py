import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, seed):
        self.state = np.random.RandomState(seed)

    def generate(self, var_type, sample_size, dim=1, low=0, high=1):
        """Generate samples of the specified type

        Parameters:§
            var_type (str): Type of the variable
            sample_size (int): Number of samples to generate
            low (int): the low boundary of the random variable
            high (int): the high boundary of the random variable

        Returns:
            A numpy array of samples drawn the specified underlying distribution
        """
        if var_type.lower() == 'cont':
            return self.state.uniform(low, high, size=(sample_size, dim))
        elif var_type.lower() == 'ord' or var_type.lower() == 'nom':
            return self.state.randint(low, high+1, size=(sample_size, dim))
        else:
            return None

    def binom(self, n, p_vector):
        return self.state.binomial(n, p_vector)

    def randn(self, sample_size, dim=1):
        return self.state.randn(sample_size, dim)


class ITRDataTable:
    """A data table of covariates, actions, and responses for ITR.

    Attributes:
        sample_size: (int or tuple) Number of samples
        n_cont: Number of continuous variables
        n_ord:  Number of ordinal variables
        n_nom:  Number of nominal variables
        n_act: Number of responses
        df:     Data frame holding the content of the table
    """

    def __init__(self, sample_size, n_act, ydim, engine):
        self.sample_size = sample_size
        self.n_act = n_act
        self.ydim = ydim
        self.engine = engine
        self.df = None
        self.x = None
        self.x_title = None
        self.act = None
        self.y = None
        self.ys = None
        self.azero = None
        
    def gen_x(self, x_func):
        self.x_title, self.x = x_func(self.sample_size, self.engine)
        
    def fillup_x(self):
        """Generate data using the provided data generator

        Parameters:

        Returns:
            None
        """
        assert not np.all(self.x == None)
        self.df = pd.DataFrame(self.x, columns=self.x_title)
    
    def gen_a(self, a_func):
        self.act = a_func(self.x, self.n_act)

    def fillup_a(self):
        """Fill up A according to the indicated model parameters (beta) and number of treatment options (n)

        :param beta: (list) The model parameter to generate A from X, should be the same dimension of X

        :return: None
        """
<<<<<<< HEAD
        self.act = a_func(self.x, self.n_act, self.engine)
        self.array = np.append(self.array, self.act, axis=1)
=======
        assert not np.all(self.act == None)
>>>>>>> 65d1dc9bf416f7d43915616d9c15bf544725fe9f
        self.df.insert(loc=0, column='Trt', value=self.act.flatten())
        
    def gen_y(self, y_func):
        assert not np.all(self.x == None)
        assert not np.all(self.act == None)
        self.y = y_func(self.x, self.act, self.ydim)
        
    def fillup_y(self):
        """

        :param y_func:
        :return:
        """
<<<<<<< HEAD
        self.y = y_func(self.x, self.act, self.ydim, self.engine)
=======
>>>>>>> 65d1dc9bf416f7d43915616d9c15bf544725fe9f
        assert self.ydim == self.y.shape[1]
        if self.y.shape[1] == 1:
            self.df.insert(loc=0, column="Y", value=self.y[:, 0])
        else:
            for i in range(self.y.shape[1]-1,-1,-1):
                self.df.insert(loc=0, column=f"Y_{i}", value=self.y[:, i])
   
    def get_testcol(self):
        if self.ydim == 1:
            return [f"Y({act})" for act in range(1, self.n_act + 1)]
        else:
            return [f"Y({act})_{ndim}" for act in range(1, self.n_act + 1) for ndim in range(self.ydim)]
    
    def gen_ys(self, y_func):
        y_matrix = np.zeros((self.sample_size, self.n_act, self.ydim))
        for trt in range(1, self.n_act + 1):
            y_matrix[:, trt - 1] = y_func(self.x,
                                               np.ones(self.sample_size).reshape(-1, 1) * trt,
                                               self.ydim)
        y_sum = np.sum(y_matrix, axis=-1)
        self.azero = np.argmax(y_sum, axis=1) + 1
        self.ys = y_matrix
        
    def fillup_ys(self):
        """
        Fill up ys of all the treatment. Used for test dataset
        """
        test_ys_df = pd.DataFrame(self.ys.reshape(self.sample_size, -1),
                                  columns=self.get_testcol())
        # test_ys_df['A'] = self.act #no need to assign act here.
        test_ys_df['A0'] = self.azero
        self.df = pd.concat([self.df, test_ys_df], axis=1)
        
    
    def export(self, fname):
        """Save the data table to the specified file name"""

        self.df.index.name = 'SubID'
        self.df.to_csv(fname)

class SimulationEngine:
    """Create training and testing tests for ITR

    Attributes:
        training_size (int): Sample size of the training data set
        testing_size (int): Sample size of the testing data set
        n_cont (int): Number of continuous variables
        n_ord (int): Number of ordinal variables
        n_nom (int): Number of nominal variables
        n_act (int): Number of responses
        training_data (ITRDataTable): Training data set
        testing_data (ITRDataTable):  Testing data set
    """

    def __init__(self, x_func, a_func, y_func, generator, n_act, ydim,
                 training_size=500, testing_size=50000):
        self.x_func = x_func
        self.a_func = a_func
        self.y_func = y_func
        self.generator = generator
        self.n_act = n_act
        self.ydim = ydim
        self.testing_size = testing_size
        self.training_data = ITRDataTable(training_size, n_act, ydim, generator)
        self.testing_data = ITRDataTable(testing_size, n_act, ydim, generator)


    def generate(self):
        """Generate training and testing data using the specified generator

        Parameters:
            generator (DataGenerator): Generator

        Returns:
            None
        """
        self.training_data.gen_x(self.x_func)
        self.training_data.gen_a(self.a_func)
        self.training_data.gen_y(self.y_func)
        
        self.testing_data.gen_x(self.x_func)
        self.testing_data.gen_ys(self.y_func)

<<<<<<< HEAD
        self.training_data.fillup_x(self.x_func)
        self.training_data.fillup_a(self.a_func)
        self.training_data.fillup_y(self.y_func)
        self.testing_data.fillup_x(self.x_func)

    def tys(self):
        """
        Generate testing ys
        :return: n x n_resp matrix
        """
        self.testing_data.fillup_a(self.a_func)
        y_matrix = np.zeros((self.testing_size, self.n_act, self.ydim))
        for trt in range(1, self.n_act + 1):
            y_matrix[:, trt - 1] = self.y_func(self.testing_data.x,
                                               np.ones(self.testing_size).reshape(-1, 1) * trt,
                                               self.ydim, self.generator)
        return y_matrix

    def azero(self, y_matrix):
        y_sum = np.sum(y_matrix, axis=-1)
        return np.argmax(y_sum, axis=1) + 1
=======
>>>>>>> 65d1dc9bf416f7d43915616d9c15bf544725fe9f

    def export(self, desc):
        """Save the training and testing data to files.

        Parameters:
            desc (str): Description of the data set

        Returns:
            None
        """
        self.training_data.fillup_x()
        self.training_data.fillup_a()
        self.training_data.fillup_y()
        self.training_data.export(desc + "_train.csv")
        
        self.testing_data.fillup_x()
        self.testing_data.export(desc + "_test_X.csv")
        
        test_ys_df = pd.DataFrame(self.testing_data.ys.reshape(self.testing_size, -1),
                                  columns=self.testing_data.get_testcol())
        #test_ys_df['A'] = self.testing_data.act
        test_ys_df['A0'] = self.testing_data.azero
        test_ys_df.to_csv(desc + "_test_Ys.csv", index_label="SubID")
        
