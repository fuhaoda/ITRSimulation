import pandas as pd
import numpy as np
from functools import partial
import sys


class DataGenerator:
    """Generate data for ITR.

    Attributes:
        m_cont: Method used to generate continuous covariate
        m_ord:  Method used to generate ordinal covariate
        m_nom:  Method used to generate nominal covariate
        m_act:  Method used to generate action 
        m_resp: Method used to generate response variable
    """

    def __init__(self, seed):
        # m_cont = partial(np.random.uniform(0, 1)),
        # m_ord = partial(np.random.randint(0, 4)),
        # m_nom = partial(np.random.randint(0, 4)),
        # m_act = partial(np.random.randint(0, 2)),
        # m_resp = partial(np.random.uniform(0, 1))):
        np.random.seed(seed)
        # self.var_type = var_type

        # self.m_cont = m_cont
        # self.m_ord = m_ord
        # self.m_nom = m_nom
        # self.m_act = m_act
        # self.m_resp = m_resp

    def generate(self, var_type, sample_size, low=0, high=1):
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
            # return self.m_cont(sample_size)
            return np.random.uniform(low, high, size=sample_size)
        elif var_type == 'ord':
            # return self.m_ord(sample_size)
            return np.random.randint(low, 4, size=sample_size)
        elif var_type == 'nom':
            # return self.m_nom(sample_size)
            return np.random.randint(low, 4, size=sample_size)
        elif var_type == 'act':
            # return self.m_act(sample_size)
            return np.random.randint(low, high, size=sample_size)
        else:
            # return self.m_resp(sample_size)
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

    def fillup_x(self):
        """Generate data using the provided data generator

        Parameters:
            # table_type (str): Type of the table, either 'training' or 'testing'
            engine (DataGenerator): Generator for the data

        Returns:
            None
        """

        for i in range(self.n_cont):
            temp = self.engine.generate('cont', self.sample_size, low=0, high=1)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['Cont' + str(i)] = temp

        for i in range(self.n_ord):
            temp = self.engine.generate('ord', self.sample_size, low=0, high=4)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['Ord' + str(i)] = temp

        for i in range(self.n_nom):
            temp = self.engine.generate('nom', self.sample_size, low=0, high=4)
            self.array = np.append(self.array, temp.reshape(-1, 1), axis=1)
            self.df['Nom' + str(i)] = temp

        return self.array

        # self.df['A'] = engine.generate('act', self.sample_size, low=0, high=self.n_resp)
        # # No. of Action should be corresponding to n_resp
        #
        # if table_type == 'training':
        #     for i in range(self.n_resp):
        #         self.df['Y' + str(i)] = engine.generate('resp', self.sample_size, low=0, high=1)
        # else:
        #     pass
    def fillup_a(self, beta):
        """Fill up A according to the indicated model parameters (beta) and number of treatment options (n)

        :param beta: (list) The model parameter to generate A from X, should be the same dimension of X

        :return: None
        """
        assert len(beta) == self.array.shape[1]
        z = np.matmul(self.array, np.array(beta).reshape(-1, 1))
        p = 1 / (1 + np.exp(-z))
        a = np.random.binomial(self.n_resp - 1, p) + 1
        self.array = np.append(self.array, a, axis=1)
        self.df['A'] = a.flatten()
        return a

    def fillup_y(self, beta, t, engine, ydim, array=None):
        """

        :param beta: (list(list)) The model parameter to generate Y from X, should be the same dimension of X
        :param t: (list(list)) The model parameter to generate Y from A, should be the same dimension of A (normally 1)
        :param array:
        :param ydim: (int) the dimension of Y
        :param engine:
        :return: y in shape (n x 1)
        """
        if array is None:
            array = self.array
        beta_t = np.append(beta, t, axis=1)
        assert beta_t.shape == (ydim, array.shape[1])

        y = np.matmul(array, beta_t.T) + \
            engine.generate('nom', (self.sample_size, ydim), low=0, high=1)

        for i in range(ydim):
            self.df['Y_'+str(i)] = y[:, i]

        return y

    # def fillup_testset(self, fillup_y):
    #     """
    #
    #     :return:
    #     """
    #     y_matrix=np.array([[]])
    #     for trt in range(1, self.n_resp+1):
    #         temparray = self.array
    #         temparray[:, -1] = trt
    #         y_matrix = np.append(y_matrix, fillup_y)




    #     def set_testing_response(self, generator):
    #         trts = {}
    #         iter = 0
    #         for v in self.action:
    #             if not v in trts:
    #                 trts[v] = iter
    #                 iter += 1

    #         nb_uniq_actions = len(set(self.action))
    #         sample_size = len(self.response)
    #         nb_response = len(self.response[0])

    #         counterfactuals = [[[0] * nb_response for i in range(sample_size)]
    #                            for j in range(nb_uniq_actions)]

    #         for i in range(nb_uniq_actions):
    #             for j in range(sample_size):
    #                 for k in range(nb_response):
    #                     counterfactuals[i][j][k] = generator()

    #         for j in range(sample_size):
    #             i = trts[self.action[j]]
    #             for k in range(nb_response):
    #                 self.response[j][k] = counterfactuals[i][j][k]

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

    def __init__(self, generator, n_cont, n_ord, n_nom, n_resp, beta_a, beta_y, t_y,
                 training_size=500, testing_size=50000):
        assert len(beta_y) == len(t_y)
        self.generator = generator
        self.beta_a = beta_a
        self.beta_y = beta_y
        self.t_y = t_y
        self.n_resp = n_resp
        self.ydim = len(beta_y)
        self.testing_size = testing_size
        self.training_data = ITRDataTable(training_size, n_cont, n_ord, n_nom, n_resp, generator)
        self.testing_data = ITRDataTable(testing_size,
                                         n_cont, n_ord, n_nom, n_resp, generator)

    def generate(self):
        """Generate training and testing data using the specified generator

        Parameters:
            generator (DataGenerator): Generator 

        Returns: 
            None
        """
        # beta_a = [-2.5, 3, 0, 1, 1]
        # beta_y = [[-2, 3, 0, 1, 1],
        #           [-2, 3, 0, 1, 1]]
        # t_y = [[0.5],
        #        [0.7]]
        self.training_data.fillup_x()
        self.training_data.fillup_a(self.beta_a)
        self.training_data.fillup_y(beta=self.beta_y, t=self.t_y, engine=self.generator, ydim=len(self.beta_y))

        # self.testing_data.export("test_X.csv")
        # self.testing_data.fillup_a(beta_a)
        # return

    def tys(self):
        testxarray = self.testing_data.fillup_x()
        a_obs = self.testing_data.fillup_a(self.beta_a)
        y_matrix = a_obs
        for trt in range(1, self.n_resp+1):
            temparray = np.append(testxarray, np.ones(self.testing_size).reshape(-1, 1) * trt, axis=1)
            y = self.testing_data.fillup_y(self.beta_y, self.t_y, self.generator, len(self.t_y), temparray)
            y_matrix = np.append(y_matrix, y, axis=1)

        return y_matrix

    def azero(self,y_matrix):
        z = y_matrix[:, 0].reshape(-1, 1)
        idx = range(y_matrix.shape[1])
        idxlist = [idx[i:i + self.ydim] for i in range(1, y_matrix.shape[1], self.ydim)]
        for i in range(len(idxlist)):
            # print(np.sum(z[:, idxlist[i]], axis=1))
            z = np.append(z, np.sum(y_matrix[:, idxlist[i]], axis=1).reshape(-1, 1), axis=1)
        return np.argmax(z[:, 1:], axis=1) + 1

    def export(self, desc):
        """Save the training and testing data to files. 

        Parameters: 
            desc (str): Description of the data set 

        Returns:
            None
        """
        self.training_data.export("train" + desc + ".csv")
        self.testing_data.export("test" + desc + ".csv")


def main():
    """
    This is only for test purpose, will be removed before release
    :return:
    """

    g = DataGenerator(seed=1)
    # t = ITRDataTable(100,2,1,1,2,g)
    # t.export("test")
    s = SimulationEngine(beta_a=[-2.5, 3, 0, 1, 1],
                         beta_y=[[-2, 3, 0, 1, 1],
                                 [-2, 3, 0, 1, 1]],
                         t_y=[[0.5],
                              [0.5]],
                         training_size=100,
                         testing_size=1000,
                         n_cont=2,
                         n_ord=1,
                         n_nom=1,
                         n_resp=2,
                         generator=g)
    s.generate()  # The naming seems confusing, data generator vs engine generate
    test_ys = s.tys()
    test_azero = s.azero(test_ys)

    pd.DataFrame(np.insert(test_ys, 1, test_azero, axis=1)).to_csv("test_Ys.csv")
    s.export("test")


if __name__ == "__main__":
    main()
