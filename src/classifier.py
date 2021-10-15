import numpy as np
from math import sqrt
from math import pi
from math import exp
import preprocessor
from numba import cuda
import pickle

# constant for use in finding gaussian probability # defined up here to reduce memory redundance
SQRT_2PI = np.float32((2*pi) ** 0.5)

class Classifier(object):

    def __init__(self):
        self.stats_by_label = {}

    def train(self, num_obs, labels, tokens):
        """
        Train the classifier on training data provided by self.data
        with labels provided by self.labels
        """
        num_files = num_obs // preprocessor.OBS_MEMORY_LIMIT
        self.labels = labels
        self.tokens = tokens
        file_prefix = '../data/feature_matrix_'

        # fencepost; compute initial stats for first set of data
        batch_num = 0
        file_num = batch_num + 1
        first_file = file_prefix + str(file_num)
        pickle_data = open(first_file, 'rb')
        data = pickle.load(pickle_data)
        # seperate this subset of data by label
        seperated_data = self.seperate_by_labels(data, batch_num)
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
            seperated_data = self.seperate_by_labels(data, batch_num)
            # extract statistics by label
            self.extract_statistics_by_label(seperated_data, batch_num)

        # free memory
        self.labels = []


    def seperate_by_labels(self, data, batch_num):
        """
        group observations by label by populating 
        a resultant dictionary where keys are labels
        and values are tfidf vectors
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
        batch_offset = batch_num * preprocessor.OBS_MEMORY_LIMIT

        for idx in range(len(data)): 
            # save label
            label = self.labels[idx + batch_offset]
            # save observation
            observation = data[idx]
            # append this observation to our list of observations for this label
            obs_by_label[label].append(observation)

        return obs_by_label


    def extract_statistics_by_label(self, data, batch_num):
        """
        calls self.create_distributions() for each label
        """

        for label, mini_batch in data.items():
            if (batch_num == 0):
                # store mean and std_dev of each token for each label
                #FIXME: turn this function into a kernel
                self.stats_by_label[label] = \
                        self.create_distributions(mini_batch)
            else:
                # since we have already generated distributions,
                # we need to update them with our new vals
                old_stats = self.stats_by_label[label]
                '''
                new_stats = np.empty(shape=self.tokens.shape, dtype=object)


                # copy data to device
                d_mini_batch = cuda.to_device(mini_batch)
                d_tokens = cuda.to_device(self.tokens)
                d_old_stats = cuda.to_device(old_stats)
                d_new_stats = cuda.to_device(new_stats)
                d_stats_by_label = cuda.to_device(self.stats_by_label)
                

                # define kernel params 
                threadsperblock = 1024
                blockspergrid = (self.tokens.size + (threadsperblock - 1)) // threadsperblock

                # call kernel to fill the new_stats array
                update_distributions[blockspergrid, threadsperblock](\
                        d_mini_batch,\
                        d_tokens,\
                        d_old_stats,\
                        d_new_stats,\
                        d_stats_by_label)

                # retreive resultant array from device
                new_stats = d_new_stats.copy_to_host()

                # free up GPU memory
                d_mini_batch.copy_to_host()
                d_tokens.copy_to_host()
                d_old_stats.copy_to_host()
                d_stats_by_label.copy_to_host()

                # persist statistics across object state
                self.stats_by_label[label] = new_stats
                '''
                self.stats_by_label[label] = \
                    self.update_distributions(mini_batch, old_stats)


    def create_distributions(self, observations):
        """
        extracts the mean and standard deviation of the tfidf weights for each
        token so that distributions may be created to find probability of a 
        label given a tfidf vector during prediction
        """
        # list of statistics for each token
        # at each index exists a 3 element tuple containing
        # (mean, std deviation, number of observations)
        statistics = []


        # go through each column
        for col in range(len(self.tokens)):

            # create list for entries of this column
            vals = []

            # FIXME: gotta be a smoother way to do this
            for row in observations:
                # add value to list 
                vals.append(row[col])
        
            # compute std dev over list if there are any nonzero weights
            sigma = np.longdouble(np.std(vals)) if any(vals) else 0.0
            # compute mean over list if there are any nonzero weights
            mean = np.longdouble(np.mean(vals)) if any(vals) else 0.0

            # add mean, std_dev, and n to statistics
            statistics.append( (mean, sigma, len(observations)) )

        return statistics



    '''
    @cuda.jit
    def update_distributions(mini_batch, tokens, old_stats, new_stats, stats_by_label):
        """
        CUDA Kernel that updates the mean, standard deviation, 
        and n of the tfidf weights

        loop being represented is the iteration through all tokens,
        such that each thread updates the distribution for a single token
        given a mini batch of observations

        """
        position = cuda.threadIdx.x

        if position < tokens.size:
            mean, sigma, n = old_stats[position]

            for tfidf_vec in mini_batch:
                # dereference tfdidf weight for the token at position
                new_weight = tfidf_vec[position]
                # update number of observations
                new_n = n + 1
                # update mean
                new_mean = np.double(((mean * n) + new_weight) / new_n)
                # update std deviation
                old_variance = sigma ** 2
                new_variance = np.double(((n * old_variance) + (new_weight - new_mean) * (new_weight - mean)) / new_n)
                # only perform sqauare root if we know variance is not 0
                new_sigma = np.double(new_variance ** 0.5) if new_variance else 0.0

                # set new stats as old stats so that next iteration of 
                # the loop can continue updating the distribution
                mean, sigma, n = (new_mean, new_sigma, new_n)

            # save updated statistics for this token
            new_stats[position] = ((mean, sigma, n))
    '''

    def update_distributions(self, observations, old_stats):
        """
        updates the mean, standard deviation, and n of the tfidf weights
        for each token given a new mini batch
        """
        new_stats = []

        for col in range(len(self.tokens)):
            # derefence stats 
            mean, sigma, n = old_stats[col]

            for row in observations:
                # dereference obs tfidfd weight for this token    
                new_weight = row[col]
                # update number of observations 
                new_n = n + 1
                # update mean
                new_mean = np.longdouble(((mean * n) + new_weight) / new_n)
                # update std dev
                old_variance = sigma ** 2
                new_variance = np.longdouble(((n * old_variance) + (new_weight - new_mean) * (new_weight - mean)) / new_n)
                # only perform square root if we know variance is not 0
                new_sigma = np.longdouble(new_variance ** 0.5) if new_variance else 0.0
                
                # set new stats as old stats so that next iteration of 
                # the loop can continue updating the distribution
                mean, sigma, n = (new_mean, new_sigma, new_n)

            # now save stats for this batch 
            new_stats.append((mean, sigma, n))
            
        return new_stats


    def predict(self, observation, possible_labels):
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
            observation(m-dim np.array): tfidf vector representing an observation we would like to classify
            the observation whose label we would like to predict

        returns
            prediction(string): label we predict for this observation
        """
        probabilities = {}

        # only compute probabilities for the server
        # designated "possible labels"
        for label in possible_labels:
            # probability of observing this label in our training set
            probability = float(self.stats_by_label[label]["num_obs"]) / len(self.feature_matrix)
            print("starting prob:" + str(probability))
            # compute probability of tfidf weight for each token given this label
            for idx in range(len(self.tokens)):
                observed_val = observation[idx]
                # grab mean and sigma for this token given this label
                mean = self.stats_by_label[label]["mean_vals"][idx]
                sigma = self.stats_by_label[label]["sigma_vals"][idx]
                # if there is enough data for a distribution to have been 
                # created
                sufficient_data = bool(sigma)
                # the token was observed in the observation we are trying
                # to classify
                token_observed = bool(observed_val)

                if sufficient_data and token_observed:
                    # calculate gaussian probability 
                    prob_of_tfidf_weight = calculate_gaussian_probability(observed_val, mean, sigma)
                    # update probability of observing this label
                    probability *= prob_of_tfidf_weight

            # double probability since we have reduced search in half
            probabilities[label] = probability * 2
            print(probabilities[label])

        # sort the probabilities by values, and return highest prob key-pair
        prediction = sorted(probabilities.items(), key=lambda x: x[1])[0]
        print(probabilities)

        return prediction

# added decorator to explicitly cast and specify cuda as runtime
def calculate_gaussian_probability(x, mean, sigma):
    """ Calculate the Gaussian probability distribution function for x
    """
    exponent = exp(-((x - mean)**2 / (2 * sigma**2)))
    return (1 / (sqrt(2 * pi) * sigma)) * exponent

