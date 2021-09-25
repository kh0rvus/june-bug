import binascii
import json
import numpy as np
import math

class Preprocessor(object):


    def __init__(self):
        self.raw_data_file = '../data/raw.json'
        self.raw_data = {}
        self.observations = []
        self.labels = []


    def collect_data(self, server, observations):
        """
        collects data by continually requesting samples from the server

        the goal is to collect:
        - the binary blob
        - the 6 possible ISAs (to narrow search space)
        - the true ISA
        This is acheived by consecutively calling get() then post() to extract the information and write it to the raw_data_file

        params:
            - observations(int): number of observations to collect
        """ 
        print("[...] collecting data")
        training_data = []
        for i in range(observations):
            # request data
            obs = self.extract_training_data(server)
            training_data.append(obs)
            print(str(i) + "requests made so far")

        # store it in the json file
        with open(self.raw_data_file, 'w') as file:
            json.dump(training_data, file)
            
        print("[+] collected data")


    def extract_training_data(self, server):
        """ 
        requests the blob, possible target data, and label for training

        params:
            - server(object): server object provided by sample code
        returns:
            - observation(object): object containing the observation label, hex blob in a string, possible ISAs in a list
        """ 
        # make arbitrary guess to get observation and label
        server.get()
        server.post('arm')

        # convert to hex for easier analysis
        hex_blob = bytes.hex(server.binary)

        # save this observation into a json entry
        hex_blob = bytes.hex(server.binary)
        observation = {
            "label": server.ans,
            "blob": hex_blob,
            "possible_ISAs": server.targets
            }

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
            self.observations.append(self.Observation(observation['blob'], observation['possible_ISAs']))
            # format label
            self.labels.append(observation['label'])

        # FIXME: delete for production
        self.observations = self.observations
        self.labels = self.labels
        print("[+] retreived data")
        return 

    def extract_tfidf(self):
        """
        creates a new matrix containing the 
        observation data plus the tfidf vector for each observation

        returns:
            feature_matrix(np.array): numpy array of the observation features
        """
        self.token_vec = set()
        feature_matrix = []

        idx = 0
        # compute term frequencies and populate token vec
        for observation in self.observations:
            print('[...] computing training term frequencies: '+ str(idx) + '/' +  str(len(self.observations)))
            term_freq = observation.compute_term_freq()
            self.token_vec.update(term_freq.keys())
            idx += 1
        print('[+] computed training term frequencies')

        # sort in token vec in alphanumeric order
        self.token_vec = sorted(list(self.token_vec))
        
        # compute idf val for each token
        self.idf_vec = []
        num_observations = len(self.observations)
        idx = 0
        for token in self.token_vec:
            blobs_with_token = 0
            for observation in self.observations:
                idx +=1
                print('[...] computing idf values for each token: '+ str(idx) + '/' +  str(len(self.observations) * len(self.token_vec)))
                # if the token appears in this observation
                if token in observation.freq_map.keys():
                    blobs_with_token += 1
            # compute idf val for this token if denominator is not 0
            idf_val = math.log(num_observations / blobs_with_token) if blobs_with_token else 0

            self.idf_vec.append(idf_val)
        
        print('[+] computed idf values for each token')
        # compute tfidf vector for each observation 
        idx = 0
        for observation in self.observations:
            idx +=1
            print('[...] computing tfidf vector for each observation:' + str(idx) + '/' + str(len(self.observations)))
            observation.tfidf_vec = self.extract_tfidf_vec(observation)
            feature_matrix.append(observation.tfidf_vec)

        print('[...] computed tfidf vector for each token')
        return np.array(feature_matrix)

    def extract_tfidf_vec(self, observation):
        """
        extracts tdfidf vector for a single observation
        """ 
        tfidf_vec = []
        # for all tokens
        for idx in range(len(self.token_vec)):
            token = self.token_vec[idx]
            idf_val = self.idf_vec[idx]

            if token in observation.freq_map.keys():
                tf_val = observation.freq_map[token]
                tfidf_vec.append(tf_val * idf_val)
            else:
                # if we dont see this token in this observation
                tfidf_vec.append(0)
        return tfidf_vec

    class Observation(object):
        def __init__(self, blob, possible_labels):
            self.blob = blob
            self.possible_labels = possible_labels
            self.freq_map = {}
            self.tfidf_vec = []
        

        def compute_term_freq(self):
            """
            creates a dict containing the tokens and term frequency
            per token for this observation

            term frequency(token) = appearances of token / number of tokens in document
            """
            tokens = self.tokenize()
            # loop through all tokens in list
            for token in tokens:
                if token in self.freq_map:
                    appearances = self.freq_map[token] * len(tokens)
                    self.freq_map[token] = (appearances + 1) / len(tokens)
                else:
                    # does not exist, add it
                    self.freq_map[token] = 1 / len(tokens)

            return self.freq_map

        def tokenize(self):
            """ 
            returns a list of all possible tokens for this observation

                word size: 1 byte
                possible token sizes: 1, 2, or 4 words long (8,16, or 32 bits)

            """ 
            all_tokens = []
            word_size = 2
            for token_size in range(2,9): 
                # ensure that we maintain instruction alignment
                if token_size % word_size == 0 and token_size != 6:
                    tokens = [self.blob[i:i+token_size] for i in range(0,len(self.blob), token_size)]
                    # add these tokens to our full token list
                    all_tokens.extend(tokens)

            return all_tokens

