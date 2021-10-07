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

        self.tokenized_obs = np.array(self.tokenized_obs)
        self.tokens = np.array(sorted(tuple(token_set)), dtype=np.uint32)
        # copy data to device
        #d_observations = cuda.const.array_like(self.tokenized_obs)
        #d_tokens = cuda.const.array(self.tokens)

        idf_vals = np.empty(shape=self.tokens.shape, dtype=np.single)
        d_idf_vals = cuda.to_device(idf_vals)

        # define kernel params 
        threadsperblock = 1024
        blockspergrid = (self.tokens.size + (threadsperblock - 1)) // threadsperblock

        print("[...] computing inverse document frequency vector")
        #idf_kernel[blockspergrid, threadsperblock](tokens, d_observations, d_idf_vals)
        #FIXME: change idf vals to idf vec
        idf_kernel[blockspergrid, threadsperblock](self.tokens, self.tokenized_obs, idf_vals)

        print("[+] computed inverse document frequency vector!")

        # free up memory
        #d_observations.copy_to_host()

        # compute tfdif vector for each observation
        for observation in self.tokenized_obs:
            # copy data to device
            tfidf_vec = np.empty(shape=self.tokens.shape, dtype=np.single)
            d_tfidf_vec = cuda.to_device(tfidf_vec)

            # flip observations, elements <> index 
            bin_count_obs = np.bincount(observation)

            # FIXME: hacky solution to not knowing how to pass scalars
            n = np.array((len(self.tokenized_obs),), dtype=np.uint32)
            # compute tfdif vector for this observation 
            tfidf_kernel[blockspergrid, threadsperblock](self.tokenized_obs, self.tokens, idf_vals, tfidf_vec, n)

            # copy data back to host
            tfidf_vec = d_tfidf_vec.copy_to_host()
            print(tfidf_vec)
            exit()
            self.feature_matrix.append(tfidf_vec)

    @cuda.jit
    def tfidf_kernel(obs, tokens, idf_vals, n, tfidf_vec):
        '''
        CUDA Kernel that computes a tfidf vector for an observation

        the loop being represented by the kernel is iteration 
        through each token such that each thread computes the 
        tfidf value for a single token for this observation

        this means that this function will need to be called 
        n times (where n is number of observations)

        params:
            - bin_count_obs(l-dim np.array): array generated from np.bincount(),
            giving an array that we can index into  using a token, and 
            retreive the number of times it occured within this observation
            - tokens(m-dim np.array): all tokens in the corpus
            - idf_vals(m-dim np.array): idf value for each token in the corpus
            - n(int): number of observations in corpus
            - tfidf_vec(m-dim np.array): output array for threads in this observation
        '''
        assert idf_vals.size == tokens.size

        # calculate position
        position = cuda.threadIdx.x

        # perform bounds checking
        if position < tokens.size:
            # dereference values
            token = tokens[position]
            idf_val = idf_vals[position]
            occurences = bin_count_obs[token]
            # FIXME: this line is kinda hacky,
            # need to look in to how to create global scalars at compile time
            term_freq = occurences / n[0]
            # set tfidf value
            tfidf_vec[position] = term_freq * idf_val
print(self.feature_matrix)
exit()


 
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
        for data_point in raw_data[:1000]:
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
        return np.array(all_tokens, dtype=np.uint32)


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


@cuda.jit
def idf_kernel(tokens, observations, idf_vals): 
    '''
    CUDA Kernel where the loop being represented is iteration through
    all tokens, to generate an inverse-document frequency value for 
    each token. thread.idx is used to index into the idf_vals array as
    well as the tokens array 

    params (device arrays):
        - tokens (m-dim np.array): each element is one of possible tokens
        - observations (n-dim*l-dim np.array): each element is an array 
        representing the tokenized observation
        - idf_vals (m-dim np.array): output device representing idf vals

    '''
    # we should have one idf val for each token
    assert idf_vals.size == tokens.size

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
            for obs_token in observation:
                if obs_token == token:
                    # increment count if token occurs in this observation
                    count += 1
                    continue

        # should have been observed at least once if it was in the token vector
        assert (count > 0)

        # calculate according to formula
        idf_vals[position] = math.log(len(observations)) / count if count else 0




