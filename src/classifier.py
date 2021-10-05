import numpy as np
from math import sqrt
from math import pi
from math import exp
import preprocessor

# constant for use in finding gaussian probability
# defined up here to reduce memory redundance
SQRT_2PI = np.float32((2*pi) ** 0.5)

class Classifier(object):

    def __init__(self, feature_matrix, labels, token_vec):
        self.observations = feature_matrix
        self.labels = labels
        self.token_vec = token_vec

        self.obs_by_label = {
                "avr":[],
                "alphaev56":[],
                "arm":[],
                "m68k":[],
                "mips":[],
                "mipsel":[],
                "powerpc":[],
                "s390":[],
                "sh4":[],
                "sparc":[],
                "x86_64":[],
                "xtensa":[]
                }
        self.stats_by_label = {}

    def train(self):
        """
        Train the classifier on training data provided by self.data
        with labels provided by self.labels
        """
        # seperate observations by label
        self.seperate_by_labels()

        # extract statistics by label
        self.extract_statistics_by_label()


    def seperate_by_labels(self):
        """
        group observations by label by populating self.classes
        """

        assert len(self.observations) == len(self.labels), 'number of \
                observations differs from number of labels'

        # labels and data is aligned, so we can iterate through 
        # observation only

        for idx in range(len(self.observations)): 
            # save label
            label = self.labels[idx]
            # save observation
            observation = self.observations[idx]
            # append this observation to our list of observations for this label
            self.obs_by_label[label].append(observation)


    def extract_statistics_by_label(self):
        """
        calls self.extract_statistic() for each label
        """

        for label, observations in self.obs_by_label.items():
            # store mean and std_dev of each token for each label
            self.stats_by_label[label] = self.extract_statistics(observations)

    def extract_statistics(self, observations):
        """
        extracts the mean and standard deviation of the tfidf weights for 
        each token
        params:
            data(2-D List): 2 dimensional list of tfidf weights where rows 
            are observations and columns are possible tokens
        """
        statistics = {
                "num_obs": len(observations),
                "mean_vals": [],
                "sigma_vals": []
                }

        idx = 0
        # go through each column
        for col in range(len(self.token_vec)):
            idx += 1
            print("[...] extracting mean and standard deviation of tfidf-weights for each token" + 
                    str(idx) + '/' + str(len(self.token_vec)))
            # create list for entries of this column
            vals = []
            # for every row in the column
            for row in observations:
                # add value to list 
                vals.append(row[col])
        
            # compute std dev over list
            sigma = np.float32(np.std(vals))
            # compute mean over list
            mean = np.float32(np.mean(vals))

            # add mean and std_dev to statistics
            statistics["mean_vals"].append(mean)
            statistics["sigma_vals"].append(sigma)

        print("[+] extracted mean and standard deviation of tfidf-weights for each token")
        # statistics should be a dictionary
        return statistics




    def predict(self, observation):
        """
        P(label|observation) = 

        if label exists in possible_ISA's list: 
        P(token_0_tfw|label) *...* P(token_m_tfw|label) * (P(label) * 2)

        else: 
        0

        probability of observation belonging to a label is the product of 
        the individual probabilities of observing the observed token 
        tfidf weights within the training set for this label multiplied 
        by the product of two and the probability of observing this label 
        in the training set, since we can reduce our search space by 50% 
        using the possible_labels given by the server

        params:
            observation(observation object): observation object containing 
            the observation whose label we would like to predict

        returns
            prediction(string): label we predict for this observation
        """
        probabilities = {}

        # only compute probabilities for the server
        # designated "possible labels"
        for label in observation.possible_labels:
            # probability of observing this label in our training set
            probability = float(self.stats_by_label[label]["num_obs"]) / len(self.observations)
            # compute probability of tfidf weight for each token given this label
            for idx in range(len(self.token_vec)):
                observed_val = observation.tfidf_vec[idx]
                # grab mean and sigma for this token given this label
                mean = self.stats_by_label[label]["mean_vals"][idx]
                sigma = self.stats_by_label[label]["sigma_vals"][idx]
                # calculate gaussian probability 
                prob_of_tfidf_weight = calculate_gaussian_probability(observed_val, mean, sigma)
                # update probability of observing this label
                probability *= prob_of_tfidf_weight

            # double probability since we have reduced search in half
            probabilities[label] = probability * 2

        # sort the probabilities by values, and return highest prob key-pair
        prediction = sorted(probabilities.items(), key=lambda x: x[1])[0]

        return prediction


# added decorator to explicitly cast and specify cuda as runtime
def calculate_gaussian_probability(x, mean, sigma):
    """ Calculate the Gaussian probability distribution function for x
    """
    return exp(-0.5 * ((x - mean) / sigma**2) / (sigma * SQRT_2PI))
