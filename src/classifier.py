class Classifier(object):

    def __init__(self, feature_matrix, labels):
        self.tokens = 
        self.data = feature_matrix
        self.labels = labels
        self.classes = {
                'avr':[],
                'alphaev56':[],
                'arm':[],
                'm68k':[], 
                'mips':[],
                'mipsel':[],
                'powerpc':[],
                's390':[],
                'sh4':[],
                'sparc':[],
                'x86_64':[],
                'xtensa':[]
                }


    def create_classes(self):
        """
        group observations by label by populating self.classes
        """

        for idx in range(len(self.labels)): 
            label = self.labels[idx]
            observation = self.data[idx]
            self.classes[label].append(observation)


    def extract_statistics(self):
        """
        extracts the mean and standard deviation of the tfidf weights for 
        each token
        """
        statistics = []
        # go through each column
            # create list for this column
            # for every row in the column
                # add value to list 
            # compute std dev over list
            # compute mean over list
            # add mean and std dev to statistics
        # convert statistics to a tuple
        pass

    def train():
        pass

    def predict():
        pass

