# number of observations in the training set
NUM_OBS = 800000

# number used to limit the amount of rows of the feature matrix
# that can be held in memory at a time. If not utilized, program will be killed
# since feature matrix is larger than amount of memory in deployment machine
OBS_MEMORY_LIMIT = 1000

# file to retreive raw data from
RAW_DATA_FILE = './data/raw.json'

# percent of documents a token must appear in in order to be identified as a 
# stop word
STOP_WORD_PERCENTILE = .99

# CUDA Threads to use per block 
THREADS = 1024
