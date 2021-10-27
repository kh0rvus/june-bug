june-bug
================

Machine Learning Multi-Class Classifier to Solve Praetorian's MLB Challenge

### Table of Contents
- [Problem Description](#problem)
- [Method](#method)
- [Drawbacks and Possible Improvements](#drawbacks)
- [Technical Documentation](#documentation)
- [Misc](#misc)

### Problem Description
----------------------------------------

> "The crux of the challenge is to build a classifier that can automatically identify and categorize the instruction set architecture of a random binary blob. Train a machine learning classifier to identify the architecture of a binary blob given a list of possible architectures. We currently support twelve architectures, including: avr, alphaev56, arm, m68k, mips, mipsel, powerpc, s390, sh4, sparc, x86_64, and xtensa."

The challenge server generates a random program and compiles it under a random instruction set architecture (ISA). 

The available architectures are provided by the wonderfully versatile cross compiler: crosstool-ng. Once compiled, the server selects a random section of instruction aligned code (this matters later). 

Given this random section of code, the challenge is to identify the original ISA from a list of possible ISAs provided by the server. 


### Method
----------------------------------

#### Data Collection
Using the provided API, my first step was to collect and format 1,000,000 data points.

As a method of formatting, I opted to store the following values within python dictionaries for each observation aggregated together in a list containing all the observations:

- `label`: extracted from `server.ans`, functions as the observation label, enabling us to train our model to produce accurate results
- `blob`: a string storing the hexadecimal representation of the binary blob given by `server.binary`

After requesting the 1,000,000 data points, the new training set is written to a local json file and the `data_collector.collect()` function can be commented out to reduce excessive traffic to the server.

#### Preprocessing
##### Tokenization
After observations and labels have been aggregated, tokenization allows us to break a binary string down into a set of tokens so that the model can use the tokens as features to draw conclusions from the data.

I decided to create tokens of the following sizes:
- 8 bits (2 hex chars)
- 16 bits (4 hex chars)
- 32 bits (8 hex chars)

I was not able to find an archetecture in the list which used only 24 bits, so to reduce memory complexity, I opted to only produce the aforementioned token sizes 

##### TF-IDF Feature Extraction
Perhaps the most important aspect of machine learning is *feature extraction*, or encoding of real world information into a format that is easily accessible by the underlying mathematical equations that govern the decisions of our agents.

In this solution, I followed the tutorial's advice and implemented the *Term-Frequency Inverse Document Frequency* feature extraction method to create a vectorized version of the training data where each row in the resultant feature matrix represents an observation with each column representing a token, thus the elements within the matrix are tfidf weights for a given row (observation) and column (token).

TF-IDF aims to extract information from tokenized text data by assigning a weight to each token of an observation that **increases** as the number of occurences of the token within said *observation* grows, but **decreases** as the number of occurences of the token within the *corpus* grows.

More specifically, tf-idf for token t can be described as:

IDF(t): ln(total number of observations / number of observations in which t appeared)

TF(t): (number of times t appears in a document) / (total number of terms in the document)

TF-IDF(t): IDF(t) * TF(t)

#### Classification 
I chose the Naive Bayes(a.k.a Idiot Bayes) Algorithm as a classifier due to its relative simplicity yet generally accurate classification capability for IID (Independent and Identically Distributed) features

Though this is an assumption that I do not have evidence to substantiate, I chose to invoke it anyways in an effort to keep the solution simple, and only to add complexity if absolutely necessary. 

Another assumption that is invoked is the distribution type; which again in an effort to keep things simple, I opted to assume that the tfidf weights for each token belong to a Gaussian Distribution

It is important to note that in stochastic processes, such as financial markets and intrusion detection, these are two very hefty claims that can negatively impact the design and ultimately the performance of machine learning agents if they do not hold in the test data-set.

Generally, the algorithm can be described as follows:
1. Seperate training data by labels
2. Compute the mean and standard deviation of tfidf weights for each tokens within a given label to produce a set of distributions that describe the tfidf weights for all observations with the same label.
3. Given a new observation, compute the probability of it belonging to each label using the following formula:
   
    P(observation | label) = P(token 0 tfidf weight| label) * ... * P(token m tfidf weight| label) * (P(label) * 2)
    
    if label exists in the given possible labels 
    
    else 
    
    P(observation| label) = 0.0

The reader may notice that the formula has been modified. Since the server helps us out by narrowing the options down to six labels as opposed to twelve, I scaled the probability of observing the label by 2 and abstained from calculating the probability at all if the label was not given as a possible label.


## Technical Documentation
--------------------------


### Hyper Parameters
the hyper parameters and constants for deployment are stored in `./src/hyper_params.py`

- `NUM_OBS`: Number of observations in the training set
- `OBS_MEMORY_LIMIT`: The maximum number of rows of the feature matrix that can be kept in memory at a time to reduce possibility of memory errors
- `RAW_DATA_FILE`: The location of file that stores the raw training data
- `STOP_WORD_PERCENTILE`: The percent of documents a token must appear in in order to be identified as stop-word
- `THREADS`: The number of CUDA threads to initialize upon deployment of GPU Kernels


### `main.py`
#### Data Structures
##### `Server(object)`
Provided object that serves as an api for communicating with the Praetorian challenge server

- `Server.session`: a `requests.session()` https session connected to the praetorian server
- `Server.binary`: the most recently requested binary blob, a string containing a 128 character hexadecimal representation of the base64-decoded binary blob produced by the server's `get()` API capability.
    - example: `0x0008815f000c7d234b787d445378397f002083ebfffc7d615b784e8000209421ffe093e1001c7c3f0b78d03f0008d05f000cc19f00083d200000c0090000ed8c`
- `Server.hash`: the flag, value is set if challenge is solved
- `Server.wins`: amount of wins during current session
- `Server.targets`: the possible labels for the most recently requested binary blob
    - example: `['arm', 'sh4', 'powerpc', 'mips', 's390', 'x86_64']`
- `Server.ans`: the true label for the most recently requested binary blob

#### Functions
##### `Server._request()`
Sends an http request to the mlb server with request type depending on the wrapper that called it.

##### `Server.get()`
Wrapper for `Server._request()` which submits a get requests a binary blob along with the 6 possible ISAs that could have produced the blob.

##### `Server.post()`
Wrapper for `Server._request()` which submits a post request with the ISA that the classifier labeled the previously recieved binary blob as. Upon response, we receive a count of current wins in `Server.wins`, the correct label in `Server.ans`, and a hash in `Server.hash` if we completed the challenge.


### `data_collector.py`
#### Functions
##### `collect(server, num_obs, data_file)`
First subroutine executed. `collect()` is responsible for receiving a number of observations to request through the `num_obs` parameter and requesting the specified amount of data using the provided `server` object before writing the data to a local file on the deployment machine whose path is provided through the `data_file` parameter.
The `collect()` function achieves this by requesting a new binary blob and submitting an arbitrary prediction (in this case 'arm') in order to compile a set of blobs and their associated labels.


### `preprocessor.py`
#### Data Structures
##### `Preprocessor(object)`
Python object utilized for preprocessing the raw data and extracting an informative feature matrix to be used by the classifier.

- `self.observations`: Numpy Array of all observations in the training set
- `self.labels`: Numpy Array of all true labels for the observations in the training set (index aligned with `self.observations`)
- `self.tokenized_obs`: Numpy Array of the tokenized representations of all observations in the training set
- `self.tokens`: Numpy Array of all tokens observed in the corpus
- `self.idf_vec`: Numpy Array of the Inverse Document Frequency weights for each document in the corpus (index aligned with `self.tokens`)

#### Functions
##### `Preprocessor.preprocess(self)`
encodes the raw data collected from the challenge server into a feature matrix which contains a TF-IDF vector for each observation.

The overall subroutine consists of:
1. Tokenization of all observations in the training set and population of `self.tokens`
2. Computation of the Inverse Document Frequency weights for each token
3. Computation of the TF-IDF weights for each token within each observation
4. Persistance of data to the `./data` directory

Due to the large amount of data used in training, a few optimizations were made within this subroutine to improve efficiency.

Upon initial implementation, in a single-threaded CPU oriented environment, execution was quite slow and to my estimate, would have taken ~470 hours to complete preprocessing. I usually utilize the already optimized scikit-learn and tensorflow subroutines, so I had never had to worry much about inefficient models but thanks to the challenge restraints, I was able to use the opportunity to learn about GPU programming and hack up a solution involving the parallelization of this subroutine using the `Numba` open source JIT compiler which allows for CUDA programming in python.

Additionaly, I recognized that though I had solved the time efficiency problem with GPU parallelization, I still had a memory issue due to the large dimensionality of the produced feature matrix. As a result, I decided to paritition the training data into mini-batches of size `OBS_MEMORY_LIMIT` and use the `pickle` module to serialize subsets of the feature_matrix and write them to files on the deployment machine to avoid keeping the entire feature matrix in memory.

##### `Preprocessor.retreive_data(self)`
populates `self.observations` and `self.labels` by opening up the raw data file whose path is provided by `RAW_DATA_FILE` and storing the raw blobs and their associated labels in the aforementioned arrays respectively.

##### `Preprocessor.tokenize(self, observation)`
recieves a binary blob in string representation through `observation` and extracts tokens of size 8, 16, and 32 bits to account for the fact the possible ISAs may have varying instruction size. Since 32 bits is the largest token size, and there shouldnt be negative values, the extracted tokens are stored as 32-bit unsigned integers and returned in a numpy array 

##### `Preprocessor.remove_stopwords(self)`
reduces the dimensionality of the feature matrix by eliminating stop words. This implementation defines stop words as tokens which appear in more than `STOP_WORDS_PERCENTILE` percent of the documents. After identifying the stop words, the token and its idf weight are removed from `self.tokens` and `self.idf_vec`

##### `Preprocessor.classification_preprocess(self, blob)`
performs preprocessing for single observations belonging to the test set, identified by the `blob` parameter. Subroutine follows `Preprocessor.preprocess()`, with the only differences being that `Preprocessor.classification_preprocess()` preprocesses a single observation, as opposed to the entire training set, and an idf vector is not computed since it has already been saved to `self.idf_vec`

##### `idf_kernel(tokens, observations, idf_vec)`
a CUDA GPU Kernel which runs in a distributed manner by allocating for each token, a thread to compute the Inverse Document Frequency weight, resulting in the execution of `m` threads where `m` is the number of tokens observed in the corpus.
By taking in the `tokens` parameter, a GPU device copy of `self.tokens`, each threads index is used to access its respective element of the `idf_vec`, which serves as the output array, and the `tokens` device array. Using `observations`, which contains the entire training set of tokenized observations, idf weights are extracted in a parallel manner for each token and stored in `idf_vec`, ultimately to be placed in `self.idf_vec` by the calling function (`Preprocessor.preprocess()`)

##### `tfidf_kernel(observation, tokens, idf_vec, tfidf_vec)`
a CUDA GPU kernel which runs in a distributed manner by allocating for each token, a thread to compute the TF-IDF weight, resulting in the execution of `m` threads where `m` is the number of tokens observed in the corpus.
By taking the `tokens` parameter, a GPU device copy of `self.tokens`, each thread index is used to access its respective element of the `idf_vec` and `tokens` device array, to distributively compute the tfidf vector for `observation`, and store it in the output array; `tfidf_vec`.


### `classifier.py`
#### Constants
`SQRT_2PI`: the square root of 2 * pi, for use in finding gaussian probabilities

#### Data Structures
##### `Classifier(object)`
Naive Bayes Classifier Implementation 
- `Classifier.stats_by_label`: Python Dictionary containing the possible ISAs as keys and a tuple containing their associated statistics for each token used to describe gaussian distributions. 
    - example: `['arm':(num_observations, [t_0-mean, ..., t_m-mean], [t_0-stddev, ..., t_m-stddev]), ..., 'x86_64':(num_observations, [t_0-mean, ..., t_m-mean], [t_0-stddev, ..., t_m-stddev])]`
- `Classifier.labels`: Numpy Array of the true labels for each observation in the training set
- `Classifier.tokens`: Numpy Array of all tokens observed in the corpus

#### Functions
##### `Classifier.train(self, labels, tokens)`
trains the classifer on OBS_MEMORY_LIMIT sized mini-batches by using the first mini batch to generate a set of `m` distributions representing each tokens tfidf weight for each label, and then updating the distributions with the remaining mini-batches.

Generally, the subroutine is as follows:
1. Seperate the data by labels
2. compute mean, standard deviation, and number of observations for each token within each set of labels to describe gaussian distributions that can be used during classification to create a probability of an observation belonging to a label.

Specifically, the training for the initial mini-batch versus the rest of the mini-batches differs due to the creation of statistics for the first mini-batch using the entire mini-batch, and the updating of statistics for the following mini-batches on an observation-wise basis.

##### `Classifier.seperate_by_labels(self, mini_batch, batch_num)`
uses the `batch_num` variable to compute the index of the label of an observation within the `mini-batch` and group all observations of this mini-batch by their respective label

##### `Classifier.extract_statistics_by_label(self, mini_batch_by_label, batch_num)`
iterates through the dictionary, `mini_batch_by_label` to extract the statistics for each tokens tfidf weight within a given label. If the batch number is 0, `self.create_distributions()` is called to initalize the distributions, else, the `update_distributions()` CUDA kernel is called to update the statistics in a distributed manner.
Ultimately, the new statistics are updated and persisted by storage within the `self.stats_by_label` Dictionary.

##### `Classifier.create_distributions(self, observations)`
extracts the mean and standard deviation of the tfidf weights for each token so that distributions may be created to find the probability of a label given a tfidf vector during test-set classification. Since this is the creation, we simply aggregate all the observed tfidf weights belonging to a given token and use Numpy's `std()` and `mean()` functions to calculate the statistics
After computing the statistics, the number of observations (`num_obs`), array of all `mean` tfidf weights, and standard deviation (`sigma`) of all tfidf weights are returned.

##### `Classifier.classify(self, observation, possible_labels)`
takes in a single binary blob referenced by `observation` as well as the possible labels for this blob stored in `possible_labels` and uses the precomputed statistics for each tokens tfidf weight seperated by label in order to compute the probability of an observation belonging to a given label using the Modified Naive Bayes Formula discussed in the *Method* section.
Ultimately, the predicted label, `prediction`, is returned to be sent to the challenge server.

##### `update_distributions(mini_batch, tokens, old_mean_vec, old_sigma_vec, old_num_obs, new_mean_vec, new_sigma_vec)`
a CUDA GPU Kernel that updates the statistics for the tfidf weights for each token within a label class.
Subroutine differs from create_distributions due to the running standard devation and mean computation which operate on each element in the mini-batch rather than the entire mini-batch. Since this introduces some overhead, the function was made into a CUDA Kernel so that `m` threads can be deployed to extract the statistics for each token in parallel.

##### `calculate_gaussian_probability(x, mean, sigma)`
receives an observed value, `x`, along with a `mean` and `sigma` and computes the probability of observing `x` under the assumption that the distribution is normally distributed.


## Misc
---------------------
- Interesting related work: [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)


