import pandas as pd
import numpy as np
from functools import partial

class DataGenerator:
    """Generate data for ITR.

    Attributes:
        m_cont: Method used to generate continuous covariate
        m_ord:  Method used to generate ordinal covariate
        m_nom:  Method used to generate nominal covariate
        m_act:  Method used to generate action 
        m_resp: Method used to generate response variable
    """
    
    def __init__(self, seed, 
                 m_cont = partial(np.random.uniform(0, 1)),
                 m_ord = partial(np.random.randint(0, 4)),
                 m_nom = partial(np.random.randint(0, 4)),
                 m_act = partial(np.random.randint(0, 2)),
                 m_resp = partial(np.random.uniform(0, 1))):
        np.random.seed(seed) 
        self.m_cont = m_cont
        self.m_ord = m_ord
        self.m_nom = m_nom
        self.m_act = m_act
        self.m_resp = m_resp

    def generate(self, var_type, sample_size): 
        """Generate samples of the specified type
        
        Parameters:
            var_type (str): Type of the variable 
            sample_size (int): Number of samples to generate

        Returns: 
            An array of samples drawn the specified underlying distribution
        """
        if (var_type == 'cont'):
            return self.m_cont(sample_size)
        elif (var_type == 'ord'):
            return self.m_ord(sample_size)
        elif (var_type == 'nom'):
            return self.m_nom(sample_size)
        elif (var_type == 'act'):
            return self.m_act(sample_size)
        else:
            return self.m_resp(sample_size)
        

class ITRDataTable:
    """A data table of covariates, actions, and responses for ITR. 

    Attributes:
        sample_size: Number of samples 
        n_cont: Number of continuous variables 
        n_ord:  Number of ordinal variables
        n_nom:  Number of nominal variables 
        n_resp: Number of responses 
        df:     Data frame holding the content of the table
    """

    def __init__(self, sample_size, n_cont, n_ord, n_nom, n_resp):
        self.sample_size = sample_size
        self.n_cont = n_cont
        self.n_ord = n_ord
        self.n_nom = n_nom
        self.n_resp = n_resp
        self.df = pd.DataFrame()


    def generate(self, table_type, engine):
        """Generate data using the provided data generator

        Parameters:
            table_type (str): Type of the table, either 'training' or 'testing'
            engine (DataGenerator): Generator for the data

        Returns:
            None
        """ 
    
        for i in range(self.n_cont):
            self.df['Cont' + str(i)] = engine.generate('cont', self.sample_size)

        for i in range(self.n_ord):
            self.df['Ord' + str(i)] = engine.generate('ord', self.sample_size)

        for i in range(self.n_nom):
            self.df['Nom' + str(i)] = engine.generate('nom', self.sample_size)

        self.df['A'] = engine.generate('act', self.sample_size)

        if table_type == 'training':
            for i in range(self.n_resp):
                self.df['Y' + str(i)] = engine.generate('resp', self.sample_size)
        else:
            pass

        
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

    def __init__(self, training_size = 500, testing_size = 50000,
                 n_cont = 1, n_ord = 1, n_nom = 1, n_resp = 1) :
        self.training_data = ITRDataTable(training_size,
                                          n_cont, n_ord, n_nom, n_resp)
        self.testing_data = ITRDataTable(testing_size,
                                         n_cont, n_ord, n_nom, n_resp) 

    def generate(self, generator):
        """Generate training and testing data using the specified generator

        Parameters:
            generator (DataGenerator): Generator 

        Returns: 
            None
        """
        self.training_data.generate('training', generator)
        self.testing_data.generate('testing', generator)

    def export(self, desc):
        """Save the training and testing data to files. 

        Parameters: 
            desc (str): Description of the data set 

        Returns:
            None
        """
        self.training_data.export("train" + desc + ".csv")
        self.testing_data.export("test" + desc + ".csv")
        
        
                                              
        

            
