import binascii
import json
import numpy as np
import math
from numba import cuda

class Preprocessor(object):


    def __init__(self):
        self.raw_data_file = '../data/raw.json'
        self.raw_data = {}
        self.observations = []
        self.labels = []

    def preprocess(self):
        """
        creates a new matrix containing the 
        observation data plus the tfidf vector for each observation

        returns:
            feature_matrix(np.array): numpy array of the observation features
        """
        self.token_vec = set()
        feature_matrix = []
        # compute term frequencies and populate token vec
        out_device = cuda.device_array(shape=(n,), dtype=np.float32) 
 
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
        # create an array within data holding all entries
        data_file = open(self.raw_data_file)
        self.raw_data = json.loads(data_file.readline())

        for observation in self.raw_data:
            self.observations.append(self.observation(observation['blob'], observation['possible_isas']))
            # format label
            self.labels.append(observation['label'])

        # fixme: delete for production
        self.observations = self.observations self.labels = self.labels
        print("[+] retreived data")
        return 

       
