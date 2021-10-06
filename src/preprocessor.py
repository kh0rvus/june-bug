import binascii
import json
import numpy as np
import math
from numba import cuda

class Preprocessor(object):


    def __init__(self):
        self.raw_data_file = '../data/raw.json'

    def preprocess(self):
        """
        creates a new matrix containing the 
        observation data plus the tfidf vector for each observation

        returns:
            feature_matrix(np.array): numpy array of the observation features
        """
        # populate self.observations, self.possible_labels, and self.true_labels
        self.retreive_data()

        # list to store preprocessed feature matrix
        feature_matrix = []
        # list to store tokenized observations
        self.tokenized_obs = []
        # set to store all tokens observed
        token_set = set()

        # take np array of observations, and tokenize them
        for observation in self.observations:
            tokenized_observation = self.tokenize(observation) # uint32 tokens
            # add to our list of tokenized observations
            self.tokenized_obs.append(tokenized_observation)
            # update our colletion of all observed tokens
            token_set.update(tokenized_observation)

        self.tokens = sorted(tuple(token_set))
        print(len(self.tokens))

        # list to represent number of observations that contain 
        # each token, indexed in same order as self.tokens
        self.occurences = np.zeros(len(self.tokens))

        
        idn = 0
        # compute idf for each token given all observations
        for obs in self.tokenized_obs:
            idn += 1
            print(str(idn) + "/" + str(len(self.tokenized_obs)))
            # turn obs into a set to reduce duplicates
            obs = set(obs)
            for token in obs: 
                # save index so we match with self.tokens
                idx = self.tokens.index(token)
                # increment number of occurences for this token
                self.occurences[idx] += 1
        
        for occurences_of_token in self.occurences:
            idf_val = math.log(len(self.observations) / occurences_of_token)
            self.idf_vals.append(idf_val)

        # compute tf-idf value for each token in each observation

        print(len(self.idf_vals))
        exit()

    @cuda.jit
    def idf_kernel(): 
        '''
        CUDA Kernel that where the loop being represented in iteration through
        all tokens, thus thread.idx is used to index into the idf_vals array as
        well as the tokens array 

        has access to the following device arrays:
        - tokens (m-dim array): each element is one of possible tokens
        - observations (n*l-dim array): each element is an array representing tokenized observation
        - idf_vals (m-dim array): output device representing idf vals
        '''
        #FIXME: debug and documentation
        # calculate position (cuda.thread.idx)
        #position = cuda.thread.idx
        token = device.tokens[position]
        # number of times token occurs in corpus
        count = 0
        # iterating through all observations to search for this token
        for obs in device.observations:
            # reduce redundancy
            obs = set(obs)
        # should have been observed at least once if it was in the token vector
        assert (count > 0)
        idf_vals[position] = math.log(len(observations)) / count if count else 0



    @cuda.jit
    def tfidf_kernel(bin_count_obs):
        '''
        CUDA Kernel that computes a tfidf vector for an observation
        using the device idf_vals array and tokens array. Loop being
        represented is through the tokens such that each thread
        computes the tfidf value for a single token within this
        observation

        the observation parameter, bin_count_obs is a numpy array generated
        from the np.bincount() function, giving an array that we can index
        to using a token, and retreive the number of times it occured within
        this observation
        '''
        # calculate position
        token = tokens[position]
        idf_val = idf_vals[position]
        occurences = bin_count_obs[token]
        # FIXME: use old preprocess to recreate formula
        tf_idf_vec = 

 
    def prediction_preprocess(self, blob, possible_labels):
        """
        performs preprocessing for prediction (test set)

        creates an observation object and 
        creates tfidf vec from the blob
        
        params:
            blob(bytes): binary blob produced by the server
            possible_labels(

        """
        print("[...] preprocessing observation for prediction")
        # convert blob to hex
        hex_blob = bytes.hex(blob)

        # FIXME: we need to put this into arrays, obs objects dont exist anymore
        # create observation
        observation = self.Observation(hex_blob, possible_labels)
        # tokenize and calculate term frequencies
        observation.compute_term_freq()
        # calculate tfidf vector
        observation.tfidf_vec = self.extract_tfidf_vec(observation)
        print("[+] preprocessed observation for prediction")
        return observation


    def retreive_data(self):
        """
        opens up the raw_data_file and turns the json string into a python object

        creates the feature matrix and label vector
        """
        
        print("[...] retreiving data")
        # create an iterable within data holding all entries
        data_file = open(self.raw_data_file)
        raw_data = json.loads(data_file.readline())

        # create our output lists; holding the binary blob, possible labels, 
        # and true label for each observation respectively
        observations = []
        possible_labels = []
        true_labels = []

        # populate our output lists
        for data_point in raw_data:
            observations.append(data_point['blob'])
            possible_labels.append(data_point['possible_ISAs'])
            true_labels.append(data_point['label'])

        # convert our lists into np arrays for use with numba
        self.observations = np.array(observations)
        self.possible_labels = np.array(possible_labels)
        self.true_labels = np.array(true_labels)
        print("[+] retreived data")

        return 

    def tokenize(self, observation): 
        """ 
        returns a list of all possible tokens for this observation,
        where each token is encoded into a 32-bit unsigned integer

        word size: 1 byte
        possible token sizes: 1, 2, or 4 words long (8,16, or 32 bits)

        """ 

        all_tokens = []
        word_size = 2
        for token_size in range(2,9): 
            # ensure that we maintain instruction alignment
            if token_size % word_size == 0 and token_size != 6:
                tokens = [int(observation[i:i+token_size], 16) for i in range(0,len(observation), token_size)]
                # add these tokens to our full token list
                all_tokens.extend(tokens)

        return np.array(all_tokens)


