import numpy as np
from math import sqrt
from math import pi
from math import exp

class Classifier(object):

    def __init__(self, feature_matrix, labels):
        self.data = feature_matrix
        self.labels = labels
        self.classes = {
                'avr':[],
                'alphaev56':[],
                'arm':[],
                'm68k':[], 
                'mips':[],
                'mipsel':[],
                'powerpc':[],
                's390':[],
                'sh4':[],
                'sparc':[],
                'x86_64':[],
                'xtensa':[]
                }
        self.stats_by_label = {}


    def create_classes(self):
        """
        group observations by label by populating self.classes
        """

        for idx in range(len(self.labels)): 
            label = self.labels[idx]
            observation = self.data[idx]
            self.classes[label].append(observation)


    def extract_statistics_by_label(self):
        """
        calls self.extract_statistics() for each label
        """
        all_statistics = {}
        for label, data in self.classes.items():
            all_statistics[label] = self.extract_statistics(data)

    def extract_statistics(self, data):
        """
        extracts the mean and standard deviation of the tfidf weights for 
        each token

        params:
            data(2-D List): 2 dimensional list of tfidf weights where rows are observations and columns are possible tokens
        """
        statistics = []

        # go through each column
        for col in range(len(data[0])):
            # create list for entries of this column
            vals = []
            # for every row in the column
            for row in data:
                # add value to list 
                vals.append(row[col])
        
            # compute std dev over list
            std = np.std(vals)
            # compute mean over list
            mean = np.mean(vals)
            # add mean, std_dev, and number of observations to statistics
            statistics.append((mean, std, len(data)))
        # convert statistics to a tuple
        statistics = tuple(statistics)
        return statistics

    def calculate_gaussian_probability(self, x, mean, std_dev):
        """ Calculate the Gaussian probability distribution function for x
        """
        exponent = exp(-((x-mean)**2 / (2 * std_dev**2 )))
        return (1 / (sqrt(2 * pi) * std_dev)) * exponent

    def train():
        pass

    def predict():
        pass

