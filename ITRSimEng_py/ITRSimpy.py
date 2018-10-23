import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, seed):
        np.random.seed(seed)

    @staticmethod
    def generate(var_type, sample_size, low=0, high=1):
        """Generate samples of the specified type

        Parameters:
            var_type (str): Type of the variable
            sample_size (int): Number of samples to generate
            low (int): the low boundary of the random variable
            high (int): the high boundary of the random variable

        Returns:
            A numpy array of samples drawn the specified underlying distribution
        """
        if var_type == 'cont':
            return np.random.uniform(low, high, size=sample_size)
        elif var_type == 'ord' or var_type == 'nom':
            return np.random.randint(low, high, size=sample_size)
        elif var_type == 'nom':
            return np.random.randint(low, high, size=sample_size)
        elif var_type == 'act':
            return np.random.randint(low, high, size=sample_size)
        else:
            return np.random.uniform(low, high, size=sample_size)


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

    def __init__(self, sample_size, n_cont, n_ord, n_nom, n_resp, engine):
        self.sample_size = sample_size
        self.n_cont = n_cont
        self.n_ord = n_ord
        self.n_nom = n_nom
        self.n_resp = n_resp
        self.engine = engine
        self.array = np.ones((sample_size, 1))
        self.df = pd.DataFrame()
        self.x = None
        self.act = None
        self.y = None

    def fillup_x(self):
        """Generate data using the provided data generator

        Parameters:

        Returns:
            None
        """

        for i in range(self.n_cont):
            temp = self.engine.generate('cont', self.sample_size, low=0, high=1)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['X_Cont' + str(i)] = temp

        for i in range(self.n_ord):
            temp = self.engine.generate('ord', self.sample_size, low=0, high=4)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['X_Ord' + str(i)] = temp

        for i in range(self.n_nom):
            temp = self.engine.generate('nom', self.sample_size, low=0, high=4)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['X_Nom' + str(i)] = temp

        self.x = self.array

    def fillup_a(self, a_func):
        """Fill up A according to the indicated model parameters (beta) and number of treatment options (n)

        :param beta: (list) The model parameter to generate A from X, should be the same dimension of X

        :return: None
        """
        self.act = a_func(self.x, self.n_resp)
        self.array = np.append(self.array, self.act, axis=1)
        self.df['A'] = self.act.flatten()

    def fillup_y(self, y_func):
        """

        :param y_func:
        :return:
        """
        self.y = y_func(self.x, self.act)
        for i in range(self.y.shape[1]):
            self.df['Y_' + str(i)] = self.y[:, i]

    def export(self, fname):
        """Save the data table to the specified file name"""

        self.df.index.name = 'ID'
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

    def __init__(self, a_func, y_func, generator, n_cont, n_ord, n_nom, n_resp, ydim,
                 training_size=500, testing_size=50000):
        self.a_func = a_func
        self.y_func = y_func
        self.generator = generator
        self.n_resp = n_resp
        self.ydim = ydim
        self.testing_size = testing_size
        self.training_data = ITRDataTable(training_size, n_cont, n_ord, n_nom, n_resp, generator)
        self.testing_data = ITRDataTable(testing_size, n_cont, n_ord, n_nom, n_resp, generator)

    def generate(self):
        """Generate training and testing data using the specified generator

        Parameters:
            generator (DataGenerator): Generator

        Returns:
            None
        """

        self.training_data.fillup_x()
        self.training_data.fillup_a(self.a_func)
        self.training_data.fillup_y(self.y_func)

        self.testing_data.fillup_x()

    def tys(self):
        self.testing_data.fillup_a(self.a_func)
        a_obs = self.testing_data.act
        y_matrix = a_obs
        for trt in range(1, self.n_resp + 1):
            y = self.y_func(self.testing_data.x, np.ones(self.testing_size).reshape(-1, 1) * trt)
            y_matrix = np.append(y_matrix, y, axis=1)

        return y_matrix

    def azero(self, y_matrix):
        z = y_matrix[:, :1]
        idx = range(y_matrix.shape[1])
        idxlist = [idx[i:i + self.ydim] for i in range(1, y_matrix.shape[1], self.ydim)]
        for i in range(len(idxlist)):
            z = np.append(z, np.sum(y_matrix[:, idxlist[i]], axis=1).reshape(-1, 1), axis=1)
        return np.argmax(z[:, 1:], axis=1) + 1

    def export(self, desc):
        """Save the training and testing data to files.

        Parameters:
            desc (str): Description of the data set

        Returns:
            None
        """
        self.training_data.export(desc + "train.csv")
        self.testing_data.export(desc + "test_X.csv")