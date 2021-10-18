import numpy as np
from math import sqrt
from math import pi
from math import exp
from math import isinf
from math import isnan
import preprocessor
from numba import cuda
import pickle
import hyper_params


class Classifier(object):

    def __init__(self):
        self.stats_by_label = {}

    def train(self, labels, tokens):
        """
        Train the classifier on mini-batches of training data retreived
        from the data directory located in the root of the
        repository

        Subroutine:
        1. extract statistics for first mini batch and create the initial
        distributions for each token by first seperating the observations by
        their labels, and then creating distributions describing the tfidf weights
        for each label
        2. Iteratively retreive the OBS_MEMORY_LIMIT mini-batches
        and seperate them by label to update tfidf statistics
        """

        num_files = hyper_params.NUM_OBS // hyper_params.OBS_MEMORY_LIMIT
        self.labels = labels
        self.tokens = tokens
        file_prefix = './data/feature_matrix_'

        # fencepost; compute initial stats for first set of data
        batch_num = 0
        file_num = batch_num + 1
        first_file = file_prefix + str(file_num)
        pickle_data = open(first_file, 'rb')
        data = pickle.load(pickle_data)
        # seperate this subset of data by label
        seperated_data = self.seperate_by_label(data, batch_num)
        # extract statistics by label for the first subset
        self.extract_statistics_by_label(seperated_data, batch_num)

        for batch_num in range(1, num_files):
            # since files are not zero based
            file_num = batch_num + 1
            # create file name
            file_name = file_prefix + str(file_num)
            # open binary file
            pickle_data = open(file_name, 'rb')
            # unpickle data
            data = pickle.load(pickle_data)
            # seperate observations by label
            seperated_data = self.seperate_by_label(data, batch_num)
            # extract statistics by label
            self.extract_statistics_by_label(seperated_data, batch_num)

        # free memory
        self.labels.clear()

    def seperate_by_label(self, mini_batch, batch_num):
        """
        group observations by label by populating 
        a resultant dictionary where keys are labels
        and values are tfidf vectors

        params:
            - mini_batch(np.array): the mini batch of training observations
            to seperate
            - batch_num(int): the batch number 
        """
        obs_by_label = {
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
        # used to get flattened index
        batch_offset = batch_num * hyper_params.OBS_MEMORY_LIMIT

        for idx in range(len(mini_batch)): 
            # save label
            label = self.labels[idx + batch_offset]
            # save observation
            observation = mini_batch[idx]
            # append this observation to our list of observations for this label
            obs_by_label[label].append(observation)

        return obs_by_label


    def extract_statistics_by_label(self, mini_batch_by_label, batch_num):
        """
        calls self.create_distributions() for the first mini_batch,
        and update_distributions() for all other mini_batches to 
        extract statistics to describe distributions for each tokens tfidf
        weight within a label class

        params:
            - mini_batch_by_label(Dictionary): dictionary containing labels as
            keys and statistics for each token's tfidf weight as values
            - batch_num(int): number of the mini batch
        """

        for label, mini_batch in mini_batch_by_label.items():
            if (batch_num == 0):
                # store mean and std_dev of each token for each label
                #FIXME: turn this function into a kernel
                self.stats_by_label[label] = \
                        self.create_distributions(mini_batch)
            else:
                # since we have already generated distributions,
                # we need to update them with our new vals
                old_num_obs, old_mean_vec, old_sigma_vec = \
                        self.stats_by_label[label]

                new_mean_vec = np.empty(shape=self.tokens.shape,
                        dtype=np.double)
                new_sigma_vec = np.empty(shape=self.tokens.shape,
                        dtype=np.double)

                # copy data to device
                d_mini_batch = cuda.to_device(mini_batch)
                d_tokens = cuda.to_device(self.tokens)
                d_old_mean_vec = cuda.to_device(old_mean_vec)
                d_old_sigma_vec = cuda.to_device(old_sigma_vec)
                d_new_mean_vec = cuda.to_device(new_mean_vec)
                d_new_sigma_vec = cuda.to_device(new_sigma_vec)

                # define kernel params 
                threadsperblock = hyper_params.THREADS
                blockspergrid = (self.tokens.size + (threadsperblock - 1)) \
                        // threadsperblock

                # call kernel to fill the new_stats array
                update_distributions[blockspergrid, threadsperblock](
                        d_mini_batch, d_tokens, d_old_mean_vec,
                        d_old_sigma_vec, old_num_obs,
                        d_new_mean_vec, d_new_sigma_vec)

                # retreive resultant array from device
                new_mean_vec = d_new_mean_vec.copy_to_host()
                new_sigma_vec = d_new_sigma_vec.copy_to_host()

                # free up GPU memory
                d_mini_batch.copy_to_host()
                d_tokens.copy_to_host()
                d_old_mean_vec.copy_to_host()
                d_old_sigma_vec.copy_to_host()

                # persist statistics across object state
                self.stats_by_label[label] = (old_num_obs + len(mini_batch), 
                        new_mean_vec, new_sigma_vec)

    def create_distributions(self, observations):
        """
        extracts the mean and standard deviation of the tfidf weights for each
        token so that distributions may be created to find probability of a 
        label given a tfidf vector during prediction

        params:
            - observations(np.array): tfidf vectors for observations belonging 
            to the same labelto compute distributions from

        returns:
            - num_obs(int): number of observations in this mini_batch
            - mean_vec(np.array): array containing the mean tfidf weights for
            each token
            - sigma_vec(np.array): array containing the standard deviation 
            of the tfidf weights for each token
        """
        mean_vec = []
        sigma_vec = [] 
        num_obs = len(observations)

        # go through each column
        for col in range(len(self.tokens)):
            # create list for entries of this column
            vals = []

            # FIXME: gotta be a smoother way to do this
            for row in observations:
                # add value to list 
                vals.append(row[col])

            # compute std dev over list if there are any nonzero weights
            sigma = np.double(np.std(vals, dtype=np.float64)) if any(vals) \
                    else 0.0
            # compute mean over list if there are any nonzero weights
            mean = np.double(np.mean(vals, dtype=np.float64)) if any(vals) \
                    else 0.0
            if isnan(mean) or isnan(sigma) or isinf(mean) or isinf(sigma):
                sigma = 0
            mean_vec.append(mean)
            sigma_vec.append(sigma)

        return num_obs, mean_vec, sigma_vec


    def classify(self, observation, possible_labels):
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
            observation(m-dim np.array): tfidf vector representing an 
            observation we would like to classify

        returns
            prediction(string): label we predict for this observation
        """
        probabilities = {}

        # only compute probabilities for the server
        # designated "possible labels"
        for label in possible_labels:
            # probability of observing this label in our training set
            probability = float(self.stats_by_label[label][0]) / hyper_params.NUM_OBS
            # compute prob of tfidf weight for each token given this label
            for idx in range(len(self.tokens)):
                observed_val = observation[idx]
                # grab mean and sigma for this token given this label
                mean = self.stats_by_label[label][1][idx]
                sigma = self.stats_by_label[label][2][idx]
                # if there is enough data for a distribution to have been 
                # created
                sufficient_data = bool(sigma) and bool(mean)
                # the token was observed in the observation we are trying
                # to classify
                token_observed = bool(observed_val)

                if sufficient_data and token_observed:
                    # calculate gaussian probability 
                    prob_of_tfidf_weight = \
                            calculate_gaussian_probability(observed_val,\
                            mean, sigma)
                    if not (isinf(prob_of_tfidf_weight) or \
                            isnan(prob_of_tfidf_weight)) and \
                            prob_of_tfidf_weight > 0:
                        # update probability of observing this label
                        probability *= prob_of_tfidf_weight

            # double probability since we have reduced search in half
            probabilities[label] = probability * 2

        # sort the probabilities by values, and return highest prob key-pair
        prediction = sorted(probabilities.items(), key=lambda x: x[1])[-1]
        print(probabilities)
        return prediction


@cuda.jit
def update_distributions(mini_batch, tokens, old_mean_vec, old_sigma_vec,
        old_num_obs, new_mean_vec, new_sigma_vec):
    """
    CUDA Kernel that updates the mean, standard deviation, 
    and number of observations of the tfidf weights

    loop being represented is the iteration through all tokens,
    such that each thread updates the distribution for a single token
    given a mini batch of observations

    params:
        - mini_batch(np.array): mini batch to use for updating the
        tfidf statistics of a specific label
        - tokens(np.array): all tokens observed in the corpus
        - old_mean_vec(np.array): the previously computed means of all tfidf
        weights for this label
        - old_sigma_vec(np.array): the previously computed standard devation of
        all tfidf weights for this label
        - old_num_obs(uint32): the old number of observations observed within 
        this label
        - new_mean_vec(np.array): serves as output array to populate with new
        mean weights
        - new_sigma_vec(np.array): serves as output array to populate with new
        standard deviation of weights
    """
    position = cuda.threadIdx.x

    if position < tokens.size:
        mean = old_mean_vec[position]
        sigma = old_sigma_vec[position]
        n = old_num_obs

        for tfidf_vec in mini_batch:
            # dereference tfdidf weight for the token at position
            new_weight = tfidf_vec[position]
            # update number of observations
            new_n = n + 1
            # update mean
            new_mean = np.float64(((mean * n) + new_weight) / new_n)
            # update std deviation
            old_variance = sigma ** 2
            new_variance = np.double(((n * old_variance) + \
                    (new_weight - new_mean) * (new_weight - mean)) / new_n)
            # only perform sqauare root if we know variance is not 0
            new_sigma = np.double(new_variance ** 0.5) if new_variance else 0.0

            # set new stats as old stats so that next iteration of 
            # the loop can continue updating the distribution
            mean, sigma, n = (new_mean, new_sigma, new_n)

        # save updated statistics for this token
        new_mean_vec[position] = mean
        new_sigma_vec[position] = sigma

def calculate_gaussian_probability(x, mean, sigma):
    """ Calculate the Gaussian probability distribution function for x

        params:
            - x(float64): observed value
            - mean(float64): mean of the training set 
            - sigma(float64): standard deviation of the training set
    """
    exponent = exp(-((x - mean)**2 / (2 * sigma**2)))
    return (1 / (sqrt(2 * pi) * sigma)) * exponent
    
