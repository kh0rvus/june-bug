import binascii
import json

class Preprocessor(object):



    def __init__(self):
        self.raw_data_file = '../data/raw.json'


    '''
    collects data by continually requesting samples from the server

    the goal is to collect:
    - the binary blob
    - the 6 possible ISAs (to narrow search space)
    - the true ISA
    This is acheived by consecutively calling get() then post() to extract the information and write it to the raw_data_file

    params:
        - observations(int): number of observations to collect
    '''
    def collect_data(self, server, observations):
        training_data = []
        for i in range(observations):
            # request data
            obs = self.curate_training_data(server)
            training_data.append(obs)

        # store it in the json file
        with open(self.raw_data_file, 'w') as file:
            json.dump(training_data, file)


    '''
    requests the blob, possible target data, and label for training

    params:
        - server(object): server object provided by sample code
    returns:
        - observation(object): object containing the observation label, hex blob in a string, possible ISAs in a list
    '''
    def curate_training_data(self, server):
        # make arbitrary guess to get observation and label
        server.get()
        server.post('arm')

        # convert to hex for easier analysis
        hex_blob = bytes.hex(server.binary)

        # save this observation into a json entry
        observation = {
            "label": server.ans,
            "blob": hex_blob,
            "possible_ISAs": server.targets
            }

        return observation


    '''
    requests the blob and possible target data from the server for the test set,
    meaning we can't submit nonsense guesses for the labels but instead will need to use 
    only this data and extracted features to make classifications

    params:
        - server(object): server object provided by sample code
    returns:
        - observation(object): object containing the hex blob in a string and the possible ISAs in a list
    '''
    def curate_test_data(server):
        # request only the blob and possible ISAs
        server.get()

        # convert to hex for easier analysis
        hex_blob = bytes.hex(server.binary)

        observation = {
            "blob": hex_blob,
            "possible_ISAs": server.targets
            }

        return observation



    '''
    extracts tdfidf val matrix for a single observation
    '''
    def extract_tfidf():
        pass


    '''
    given a hex blob of data, returns a list of all possible tokens

        word size: 1 byte
        possible token sizes: 1, 2, or 4 words long (8,16, or 32 bits)

    '''
    def tokenize(self, blob):
        all_tokens = []
        word_size = 2
        for token_size in range(2,9): 
            # ensure that we maintain instruction alignment
            if token_size % word_size == 0 and token_size != 6:
                tokens = [blob[i:i+token_size] for i in range(0,len(blob), token_size)]
                # add these tokens to our full token list
                all_tokens.extend(tokens)


    
    '''
    given a hex blob and a list of tokens, return a dict containing the tokens and number of occurences per token

    params:
        tokens(list): list of tokens from a single blob

    returns:
        freq_map(dict): keys=token, values=number of times token appeared
    '''
    def term_freq(self, blob, tokens):
        freq_map = {}

        # loop through all tokens in list
        for token in tokens:
            if token in freq_map:
                freq_map[token] += 1
            else:
                # does not exist, add it
                freq_map[token] = 1

        return freq_map
        


    '''
    creates token count matrix to be used for tfidf where the rows
    represent individual blobs and the columns represent tokens
    '''  
    def count_vectorize():
        # set collection, to eliminate element redundancy
        columns = set()

        # create columns by combining tokens of all blobs into one set
        for freq_map in self.freq_maps:
            columns.add(freq_map.keys())
        columns = list(columns)
        
        data = []
        # iterate through observations to
        # create key-value pairs in each frequency
        # map such that the values are zero for tokens
        # that dont exist in the frequency map
        # FIXME: is this by reference or by value? (freq_map getting updated?)
        for freq_map in self.freq_maps:
            for column in columns:
                if column in freq_map:
                    continue
                else:
                    freq_map[column] = 0
            # sort freq maps 
            freq_map = sorted(freq_map)

        
        # at this point, all freq_maps should contain keys for every token
        # in the corpus, and 0 values for those they dont contain
        print(self.freq_maps)
        # sort all freq maps, and combine into a dataframe
        


    '''
    '''
    def create_tf_dict(self):
        pass

    def inverse_doc_freq(self):
        pass


    '''
    opens up the raw_data_file and turns the json string into a python object

    creates the feature matrix and label vector
    '''
    def retreive_data(self):
        observations = [] 
        labels = []
        
        data_file = open(self.raw_data_file)
        for line in data_file:
            obj = json.loads(line)
            print(obj)


