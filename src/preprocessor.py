import binascii
import json
import numpy as np
import math
from numba import cuda
import pickle
import main
import hyper_params



class Preprocessor(object):


    def __init__(self):
        # populate self.observations and self.true_labels
        self.retreive_data()

        # list to store tokenized observations
        self.tokenized_obs = []

    def preprocess(self):
        """
        Creates a feature matrix for the training set that contains the tfidf
        vectors for each observation. In order to avoid memory errors, the 
        training set is divided into mini-batches of size OBS_MEMORY_LIMIT.

        Every mini-batch is preprocessed seperately and written to a file in
        ../data/ in the form feature_matrix_<x> where x is the number of the 
        mini-batch. 

        The overall preprocessing subroutine consists of:
        1. Tokenized all observations and populating self.tokens with all 
        observed tokens from the training set
        2. Computing the inverse document frequency weights for each token
        3. Computing the term-frequency-inverse document frequency weights
        for each token within each observation
        """
        print("[...] Preprocessing Training Data")
        # list to store preprocessed feature matrix
        feature_matrix = []
        # set to store all tokens observed
        token_set = set()
        
        # take np array of observations, and tokenize them
        for observation in self.observations:
            tokenized_observation = self.tokenize(observation) # uint32 tokens
            # add to our list of tokenized observations
            self.tokenized_obs.append(tokenized_observation)
            # update our colletion of all observed tokens
            token_set.update(tokenized_observation)

        # free up memory
        self.observations.clear()
        
        # prepare arrays for GPU preprocessing
        self.tokenized_obs = np.array(self.tokenized_obs)
        self.tokens = np.array(sorted(tuple(token_set)), dtype=np.uint32)
        self.idf_vec = np.empty(shape=self.tokens.shape, dtype=np.double)

        # copy data to GPU
        d_tokenized_obs = cuda.to_device(self.tokenized_obs)
        d_tokens = cuda.to_device(self.tokens)
        d_idf_vec = cuda.to_device(self.idf_vec)

        # define kernel params 
        threadsperblock = hyper_params.THREADS
        blockspergrid = ((self.tokens.size + (threadsperblock - 1)) // 
                threadsperblock)

        # deploy CUDA kernel
        idf_kernel[blockspergrid, threadsperblock](d_tokens, d_tokenized_obs,
                d_idf_vec)

        # bring data back to host machine
        self.idf_vec = d_idf_vec.copy_to_host()
        self.tokens = d_tokens.copy_to_host()

        # free up memory
        d_tokenized_obs.copy_to_host()

        # remove tokens that occur in over 99% of observations
        self.remove_stopwords()

        # copy data to GPU
        d_idf_vec = cuda.to_device(idf_vec)
        d_tokens = cuda.to_device(self.tokens)
        
        # compute tfdif vector for each observation
        for observation in self.tokenized_obs:
            # copy data to device
            tfidf_vec = np.empty(shape=self.tokens.shape, dtype=np.double)
            d_tfidf_vec = cuda.to_device(tfidf_vec)
            d_observation = cuda.to_device(observation)

            # deploy CUDA kernel
            tfidf_kernel[blockspergrid, threadsperblock](d_observation, 
                    d_tokens, d_idf_vec, d_tfidf_vec)

            # copy data back to host
            tfidf_vec = d_tfidf_vec.copy_to_host()
            # free up memory
            d_observation.copy_to_host()

            # save tfidf vector and copy to file if necessary
            feature_matrix.append(tfidf_vec)
            if idx % OBS_MEMORY_LIMIT == 0:
                # too much data in memory, gotta free some up
                file_name = '../data/feature_matrix_' + str(idx // 
                        hyper_params.OBS_MEMORY_LIMIT)
                # throw into file as bytes
                with open (file_name, 'wb') as file:
                    pickle.dump(feature_matrix, file)
                # make new list for next OBS_MEMORY_LIMIT observations
                feature_matrix.clear()

        # free up GPU memory
        d_tokens.copy_to_host()
        self.idf_vec = d_idf_vec.copy_to_host()
        print("[+] Preprocessed Training Data")

    def retreive_data(self):
        """
        opens up the RAW_DATA_FILE and turns the json string into
        numpy arrays storing the data and label for all observations of
        the training set in self.observations and self.labels, respectively.
        """
        print("[...] Retreiving Training Data")
        # create an iterable within raw_data holding all entries
        data_file = open(hyper_params.RAW_DATA_FILE)
        raw_data = json.loads(data_file.readline())
        observations = []
        labels = []

        # populate our output lists
        for data_point in raw_data:
            observations.append(data_point['blob'])
            labels.append(data_point['label'])

        # convert our lists into np arrays for use with numba
        self.observations = np.array(observations)
        self.labels = np.array(labels)
        print("[+] Retreived Training Data")

    def tokenize(self, observation): 
        """ 
        returns a numpy array of all extracted tokens for this observation,
        where each token is encoded into a 32-bit unsigned integer

        word size: 1 byte
        possible token sizes: 1, 2, or 4 words long (8,16, or 32 bits)
        
        params:
            - observation(string): a binary blob produced by the challenge server

        returns:
            - numpy array of all the tokens extracted from this observation
        """ 
        all_tokens = []
        word_size = 2
        for token_size in range(2,9): 
            # ensure that we maintain instruction alignment
            if token_size % word_size == 0 and token_size != 6:
                tokens = [int(observation[i:i+token_size], 16) for i in 
                        range(0,len(observation), token_size)]
                # add these tokens to our full token list
                all_tokens.extend(tokens)

        return np.array(all_tokens, dtype=np.uint32)

    def remove_stopwords(self):
        """
        reduces the search space by eliminating tokens
        which appear in more than STOP_WORDS_PERCENTILE percent of documents
        """
        # set idf weight limit to identify stopwords
        lower_lim = math.log(hyper_params.NUM_OBS / (hyper_params.NUM_OBS * hyper_params.STOP_WORD_PERCENTILE))
        # indices of stopwords we want to remove from our list of tokens
        vals_to_delete = []

        for idx in range(len(self.idf_vec)):
            # if the token appears i
            is_stop_word = self.idf_vec[idx] < lower_lim
            if is_stop_word:
                vals_to_delete.append(idx)

        self.tokens = np.delete(self.tokens, vals_to_delete)
        self.idf_vec = np.delete(idf_vec, vals_to_delete)

    def classification_preprocess(self, blob):
        """
        performs preprocessing for observations belonging to the test set

        subroutine is identical to self.preprocess() with only differences
        being preprocessing of a single observation as opposed to the entire
        training set, as well as a lack of idf_vec creation
        
        params:
            blob(bytes): binary blob produced by the server
        """

        print("[...] Preprocessing Test Set Observation")

        # convert blob to hex
        hex_blob = bytes.hex(blob)
        # tokenize
        observation = self.tokenize(hex_blob)

        # set kernel params
        threadsperblock = hyper_params.THREADS
        blockspergrid = ((self.tokens.size + (threadsperblock - 1)) // 
                threadsperblock)

        # copy data from host to device
        d_observation = cuda.to_device(observation)
        d_tokens = cuda.to_device(self.tokens)
        d_idf_vec = cuda.to_device(self.idf_vec)

        # create output array
        tfidf_vec = np.empty(shape=self.tokens.shape, dtype=np.double)
        d_tfidf_vec = cuda.to_device(tfidf_vec)

        # calculate tfidf vector
        tfidf_kernel[blockspergrid, threadsperblock](d_observation, d_tokens,
                d_idf_vec, d_tfidf_vec)

        # copy data back to host 
        tfidf_vec = d_tfidf_vec.copy_to_host()

        # free GPU memory
        d_tokens.copy_to_host()
        d_idf_vec.copy_to_host()
        d_observation.copy_to_host()

        print("[+] Preprocessed Test Set Observation")
        return tfidf_vec


@cuda.jit
def idf_kernel(tokens, observations, idf_vec): 
    '''
    CUDA Kernel where the loop being represented is iteration through
    all tokens, to generate an inverse-document frequency value for 
    each token. thread.idx is used to index into the idf_vec array as
    well as the tokens array 

    params (device arrays):
        - tokens (m-dim np.array): each element is one of possible tokens
        - observations (n-dim*l-dim np.array): each element is an array 
        representing the tokenized observation
        - idf_vec (m-dim np.array): output device representing idf weights
        for each token
    '''
    # we should have one idf val for each token
    assert idf_vec.size == tokens.size

    # calculate position 
    position = cuda.threadIdx.x

    # perform bounds checking to account for misalignment
    # of block size and number of tokens
    if position < tokens.size:

        # access token this thread should calculate idf val for
        token = tokens[position]
        # number of times token occurs in corpus
        count = 0

        # iterating through all observations to search for this token
        for observation in observations:
            # FIXME: should be able to just remove redundancy of elements
            token_not_found = True
            idx = 0
            while token_not_found and idx < len(observation):
                obs_token = observation[idx]
                if obs_token == token:
                    # increment count if token occurs in this observation
                    count += 1
                    token_not_found = False
                idx+= 1

        # should have been observed at least once if it was in the token vector
        assert (count > 0)

        # calculate according to formula
        idf_vec[position] = (math.log(hyper_params.NUM_OBS) / count))


@cuda.jit
def tfidf_kernel(observation, tokens, idf_vec, tfidf_vec):
    '''
    CUDA Kernel that computes a tfidf vector for an observation

    the loop being represented by the kernel is iteration 
    through each token such that each thread computes the 
    tfidf value for a single token for this observation

    this means that this function will need to be called 
    n times (where n is number of observations)

    params:
        - observation(np.array): the tokenized representation of a single 
        training observation
        - tokens(np.array): all tokens observed in the corpus
        - idf_vec(np.array): idf weight for each token in the corpus
        - tfidf_vec(np.array): output array for threads in this observation
    '''
    assert idf_vec.size == tokens.size

    # calculate position
    position = cuda.threadIdx.x

    # perform bounds checking
    if position < tokens.size:
        # dereference values
        token = tokens[position]
        idf_weight = idf_vec[position]
        # find number of occurences of this token in this observation
        occurences = 0
        for obs_token in observation:
            if obs_token == token:
                occurences += 1

        # compute term frequency
        term_freq = occurences / hyper_params.NUM_OBS

        # compute tfidf weight
        tfidf_vec[position] = term_freq * idf_weight

