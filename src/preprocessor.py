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

        for _ in range(observations):
            
            # request data and give arbitrary guess
            server.get() 
            server.post('arm')

            # convert to hex for easier analysis
            hex_blob = bytes.hex(server.binary)

            # save this observation into a json entry
            obs = {
                    "label": server.ans,
                    "blob": hex_blob,
                    "possible_ISAs": server.targets
                    }
            print(obs)

            with open(self.raw_data_file, 'a') as file:
                json.dump(obs, file)

