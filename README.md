june-bug
================

Machine Learning Multi-Class Classifier to Solve Praetorian's MLB Challenge

### The Problem

> "The crux of the challenge is to build a classifier that can automatically identify and categorize the instruction set architecture of a random binary blob. Train a machine learning classifier to identify the architecture of a binary blob given a list of possible architectures. We currently support twelve architectures, including: avr, alphaev56, arm, m68k, mips, mipsel, powerpc, s390, sh4, sparc, x86_64, and xtensa."

> \- Challenge Description 



### The Oracle
The `post(target)` method takes in a guess, and returns the streak, **correct answer**, as well as a hash used later to prove authenticity of solution. This is important because the correct answer allows us to avoid the problem of manual observation labeling, largely increasing the data we can feasibly equip our model with


### Possible Labels
- avr: 8-bit
- alphaev56: 
- arm: 32/64-bit
- m68k: 16/32-bit
- mips: 32/64-bit
- mipsel: 32/64-bit
- powerpc: 32-bit
- s390: 32-bit
- sh4: 32-bit
- sparc: 32/64-bit
- x86_64: 64-bit
- xtensa: 16/24-bit


## Method

#### Data Curation
Using the provided API, my first step was to collect and format 300,000 data points.

As a method of formatting, I opted to store the following values within python dictionaries for each observation aggregated together in a list containing all the observations:

- `label`: extracted from `server.ans`, functions as the observation label, enabling us to train our model to produce accurate results
- `blob`: a string storing the hexadecimal representation of the binary blob given by `server.binary`
- `possible_ISAs`: the list of six possible ISAs that the binary blob could have been produced by, given by `server.targets`

After requesting the 300,000 data points, the new training set is written to a local json file and the `data_curation()` function can be commented out to reduce excessive traffic to the server.


#### Tokenization 
After observations and labels have been made available, tokenization must occur. 

*Tokenization* can be thought of as a process by which a document is decomposed into a set of components such that the components may be individually analyzed and used to extract new features from the data.

Based upon a few minuts spent on wikipedia finding the instruction size for each of the 12 architectures, I decided to create tokens of the following sizes:
- 8 bits (2 hex chars)
- 16 bits (4 hex chars)
- 32 bits (8 hex chars)

I was not able to find an archetecture in the list which used only 24 bits, so to reduce memory complexity, I opted to only produce the aforementioned token sizes 

#### Feature Extraction 
Perhaps the most important aspect of machine learning is *feature extraction*, or encoding of real world information into a format that is easily accessible by the underlying mathematical equations that govern the decisions of our agents.

In this solution, I followed the tutorial's advice and implemented the *Term-Frequency Inverse Document Frequency* feature extraction method to create a vectorized version of the training data. This vectorized data structure can be visualized in the *Code Documentation* section.

TF-IDF aims to extract information from tokenized text data by assigning a weight to each token of an observation that **increases** as the number of occurences of the token within said *observation* grows, but **decreases** as the number of occurences of the token within the *corpus* grows.

More specifically, tf-idf for token t can be described as:

FIXME: insert tfidf equation

#### Classification 
I chose the Naive Bayes(a.k.a Idiot Bayes) Algorithm as a classifier due to its relative simplicity yet generally accurate classification capability for IID (Independent and Identically Distributed) features

Though this is an assumption that I do not have evidence to substantiate, I chose to invoke it anyways in an effort to keep the solution simple, and only to add complexity if absolutely necessary. 

Another assumption that is invoked is the distribution type; which again in an effort to keep things simple, I opted to assume that the tfidf weights for each token belong to a Gaussian Distribution


##### Training 
##### Testing

## Optimization
### GPU-Enabled Parallel Extraction of Features

## Code Documentation

### PreProcessor
#### Data Structures
- `observation`
    - `term_freq()`:
        - returns a dictionary containing the term frequency for each token in a [[given]] observation
#### Functions

### Classifier
#### Data Structures
#### Functions

### Feature Matrix

- N = observations
- M = number of columns in the feature matrix 
- &Theta; = number of tokens observed in corpus
- &psi; = tokens_vector

the feature matrix is stored as a multi-dimensional NumPy Array that can be visualized as:


| Index   | TF-IDF Vector                       |
| ------- | ------------                        |
| 0       | [&psi;_0,&psi;_1,..,&psi;_&theta;]  |
| 1       | [&psi;_0,&psi;_1,...,&psi;_&theta;] |
| ...     | ...                                 |
| n       | [&psi;_0,&psi;_1,...,&psi;_&theta;] |

#### Blob
The **Blob** is of type string containing a 128 character hexadecimal representation of the base64-decoded binary blob produced by the server's `get()` API capability.

example:
`0x0008815f000c7d234b787d445378397f002083ebfffc7d615b784e8000209421ffe093e1001c7c3f0b78d03f0008d05f000cc19f00083d200000c0090000ed8c`

#### Possible Label Vector
The **Possible Label Vector** is a single-dimension NumPy array containing the six possible labels produced by the server's `get()` API capability.

example: 
`['arm', 'sh4', 'powerpc', 'mips', 's390', 'x86_64']`

#### TF-IDF Vector
The **TF-IDF Vector** is a single-dimension NumPy array of length &Theta; 

At each element i_&theta; of the array, exists a tf-idf weight corresponding to the token at index &theta; of the *Tokens Vector*

example: 
TODO throw example

##### The `tokens_vector`
Represented in feature matrix depiction as &psi;

Alphabetically-sorted single-dimension NumPy array containg all tokens for a training corpus with length denoted by &Theta;


example: 
TODO throw an example here



### Misc
- My cookie: 0ccbc0c1-e093-4329-9fea-76f78f2b076c
- [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)
- 


