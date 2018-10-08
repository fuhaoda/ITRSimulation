import random
import csv 

class ITRDataTable:
    def __init__(self, sample_size, nb_continuous, nb_ordinal,
                 nb_nominal, nb_response) :
        self.continuous = [[0] * nb_continuous for _ in range(sample_size)]
        self.ordinal = [[0] * nb_ordinal for _ in range(sample_size)]
        self.nominal = [[0] * nb_nominal for _ in range(sample_size)]
        self.action = [0] * sample_size
        self.response = [[0] * nb_response for _ in range(sample_size)]

    def set_continuous(self, generator):
        sample_size = len(self.continuous)
        nb_continuous = len(self.continuous[0])
        for i in range(sample_size):
            for j in range(nb_continuous):
                self.continuous[i][j] = generator()

    def set_ordinal(self, generator):
        sample_size = len(self.ordinal)
        nb_ordinal = len(self.ordinal[0])
        for i in range(sample_size):
            for j in range(nb_ordinal):
                self.ordinal[i][j] = generator()
                
    def set_nominal(self, generator):
        sample_size = len(self.nominal)
        nb_nominal = len(self.nominal[0])
        for i in range(sample_size):
            for j in range(nb_nominal):
                self.nominal[i][j] = generator()

    def set_action(self, generator):
        sample_size = len(self.action)
        for i in range(sample_size):
            self.action[i] = generator()

    def set_training_response(self, generator):
        sample_size = len(self.response)
        nb_response = len(self.response[0])
        for i in range(sample_size):
            for j in range(nb_response):
                self.response[i][j] = generator()

    def set_testing_response(self, generator):
        trts = {}
        iter = 0
        for v in self.action:
            if not v in trts:
                trts[v] = iter
                iter += 1
        
        nb_uniq_actions = len(set(self.action))
        sample_size = len(self.response)
        nb_response = len(self.response[0])

        counterfactuals = [[[0] * nb_response for i in range(sample_size)]
                           for j in range(nb_uniq_actions)]

        for i in range(nb_uniq_actions):
            for j in range(sample_size):
                for k in range(nb_response):
                    counterfactuals[i][j][k] = generator()

        for j in range(sample_size):
            i = trts[self.action[j]] 
            for k in range(nb_response):
                self.response[j][k] = counterfactuals[i][j][k]
            
    def export(self, fname):
        sample_size = len(self.continuous)
        nb_continuous = len(self.continuous[0])
        nb_ordinal = len(self.ordinal[0])
        nb_nominal = len(self.nominal[0])
        nb_resp = len(self.response[0])

        print(fname)
        with open(fname, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')

            # Write header to the file 
            row = ['ID']
            for i in range(nb_continuous):
                row.append('Cont' + str(i + 1))

            for i in range(nb_ordinal):
                row.append('Ord' + str(i + 1))

            for i in range(nb_nominal):
                row.append('Nom' + str(i + 1))

            row.append('A')

            for i in range(nb_resp):
                row.append('Y' + str(i + 1))

            writer.writerow(row)

            # Write data to the file 
            for i in range(sample_size):
                row = [str(i + 1)]

                for j in range(nb_continuous):
                    row.append(str(self.continuous[i][j]))

                for j in range(nb_ordinal):
                    row.append(str(self.ordinal[i][j]))

                for j in range(nb_nominal):
                    row.append(str(self.nominal[i][j]))

                row.append(str(self.action[i]))

                for j in range(nb_resp):
                    row.append(str(self.response[i][j]))

                writer.writerow(row)

class SimulationEngine:
    def __init__(self, **args):
        # Default values
        training_size = 5
        testing_size = 10
        nb_continuous = 1
        nb_ordinal = 1
        nb_nominal = 1
        nb_response = 1

        for k, v in args.items():
            if k == 'seed':
                random.seed(v)
            if k == 'training_size':
                training_size = v
            if k == 'testing_size':
                testing_size = v
            if k == 'nb_continuous':
                nb_continuous = v
            if k == 'nb_ordinal':
                nb_ordinal = v
            if k == 'nb_nominal':
                nb_nominal = v
            if k == 'nb_response':
                nb_response = v

        self.training_data = ITRDataTable(training_size, nb_continuous,
                                          nb_ordinal, nb_nominal, nb_response)

        self.testing_data = ITRDataTable(testing_size, nb_continuous,
                                         nb_ordinal, nb_nominal, nb_response)

    def generate(self, **args):
        for label, generator in args.items():   
            if label == "continuous":
                self.training_data.set_continuous(generator)
                self.testing_data.set_continuous(generator)
            if label == "ordinal":
                self.training_data.set_ordinal(generator)
                self.testing_data.set_ordinal(generator)
            if label == "nominal":
                self.training_data.set_nominal(generator)
                self.testing_data.set_nominal(generator)
            if label == "action":
                self.training_data.set_action(generator)
                self.testing_data.set_action(generator)
            if label == "response":
                self.training_data.set_training_response(generator)
                self.testing_data.set_testing_response(generator)

    def export(self, dataset_id):
        self.training_data.export("train" + str(dataset_id) + ".csv")
        self.testing_data.export("test" + str(dataset_id) + ".csv")
        
