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
        # elif var_type.lower() == 'act':
        #     return self.state.randint(low, high+1, size=(sample_size, dim))
        else:
            return None


class ITRDataTable:
    """A data table of covariates, actions, and responses for ITR.

    Attributes:
        sample_size: (int or tuple) Number of samples
        n_cont: Number of continuous variables
        n_ord:  Number of ordinal variables
        n_nom:  Number of nominal variables
        n_resp: Number of responses
        df:     Data frame holding the content of the table
    """

    def __init__(self, sample_size, n_resp, ydim, engine):
        self.sample_size = sample_size
        self.n_resp = n_resp
        self.ydim = ydim
        self.engine = engine
        self.array = np.ones((sample_size, 1))
        self.df = None
        self.x = None
        self.act = None
        self.y = None

    def fillup_x(self, x_func):
        """Generate data using the provided data generator

        Parameters:

        Returns:
            None
        """
        x_title, self.x = x_func(self.sample_size, self.engine)
        self.df = pd.DataFrame(self.x, columns=x_title)
        self.array = self.x

    def fillup_a(self, a_func):
        """Fill up A according to the indicated model parameters (beta) and number of treatment options (n)

        :param beta: (list) The model parameter to generate A from X, should be the same dimension of X

        :return: None
        """
        self.act = a_func(self.x, self.n_resp)
        self.array = np.append(self.array, self.act, axis=1)
        self.df.insert(loc=0, column='Trt', value=self.act.flatten())

    def fillup_y(self, y_func):
        """

        :param y_func:
        :return:
        """
        self.y = y_func(self.x, self.act, self.ydim)
        if self.y.shape[1] == 1:
            self.df.insert(loc=0, column="Y", value=self.y[:, 0])
        else:
            for i in range(self.y.shape[1]):
                self.df.insert(loc=0, column=f"Y_{i}", value=self.y[:, i])

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
        n_resp (int): Number of responses
        training_data (ITRDataTable): Training data set
        testing_data (ITRDataTable):  Testing data set
    """

    def __init__(self, x_func, a_func, y_func, generator, n_resp, ydim,
                 training_size=500, testing_size=50000):
        self.x_func = x_func
        self.a_func = a_func
        self.y_func = y_func
        self.generator = generator
        self.n_resp = n_resp
        self.ydim = ydim
        self.testing_size = testing_size
        self.training_data = ITRDataTable(training_size, n_resp, ydim, generator)
        self.testing_data = ITRDataTable(testing_size, n_resp, ydim, generator)

    def get_testcol(self):
        if self.ydim == 1:
            return [f"Y({resp})" for resp in range(1, self.n_resp + 1)]
        else:
            return [f"Y({resp})_{ndim}" for resp in range(1, self.n_resp + 1) for ndim in range(self.ydim)]

    def generate(self):
        """Generate training and testing data using the specified generator

        Parameters:
            generator (DataGenerator): Generator

        Returns:
            None
        """

        self.training_data.fillup_x(self.x_func)
        self.training_data.fillup_a(self.a_func)
        self.training_data.fillup_y(self.y_func)
        self.testing_data.fillup_x(self.x_func)

    def tys(self):
        self.testing_data.fillup_a(self.a_func)
        y_matrix = np.zeros((self.testing_size, self.n_resp, self.ydim))
        for trt in range(1, self.n_resp + 1):
            y_matrix[:, trt - 1] = self.y_func(self.testing_data.x,
                                               np.ones(self.testing_size).reshape(-1, 1) * trt,
                                               self.ydim)
        return y_matrix

    def azero(self, y_matrix):
        y_sum = np.sum(y_matrix, axis=-1)
        return np.argmax(y_sum, axis=1) + 1

    def export(self, desc):
        """Save the training and testing data to files.

        Parameters:
            desc (str): Description of the data set

        Returns:
            None
        """
        self.training_data.export(desc + "_train.csv")
        self.testing_data.export(desc + "_test_X.csv")
