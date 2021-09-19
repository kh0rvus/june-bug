june-bug
================

Machine Learning Multi-Class Classifier to Solve Praetorian's MLB Challenge

### The Problem

> "The crux of the challenge is to build a classifier that can automatically identify and categorize the instruction set architecture of a random binary blob. Train a machine learning classifier to identify the architecture of a binary blob given a list of possible architectures. We currently support twelve architectures, including: avr, alphaev56, arm, m68k, mips, mipsel, powerpc, s390, sh4, sparc, x86_64, and xtensa."
> - Challenge Description 



### Features of Interest

- The binary is instruction aligned, meaning the features we select will need to consider instruction size, rather than analyzing each hex digit, we want to be able to tell our agent to analyze the instructions
- 

### The Oracle
The `post(target)` method takes in a guess, and returns the streak, **correct answer**, as well as a hash used later to prove authenticity of solution. This is important because the correct answer allows us to avoid the problem of manual observation labeling, largely increasing the data we can efficiently equip our model with


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


## Data Structures

### Feature Matrix

- N = observations
- M = number of columns in the feature matrix 
- &Theta; = number of tokens observed in corpus
- &psi; = tokens_vector

the feature matrix is stored as a multi-dimensional NumPy Array that can be visualized as:


| Index   | Blob      | Possible Label Vector  | TF-IDF Vector                     |
| ------- | ----      | ---------------        | ------------                      |
| 0       | 0x01...23 | ['arm', ..., 'x86_64'] | [&psi_0,&psi_1,...,&psi_&theta;] |
| 1       | 0x4f...e4 | ['sparc',...,'sh4']    | [&psi_0,&psi_1,...,&psi_&theta;] |
| ...     | 0x63...3a | ['s390',...,'xtensa']  | [&psi_0,&psi_1,...,&psi_&theta;] |
| n       | 0x94...13 | ['powerpc',...,'mips'] | [&psi_0,&psi_1,...,&psi_&theta;] |

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

At each element i_&theta; of the array, exists a tf-idf statistic corresponding to the token at index &theta; of the *Tokens Vector*

example: 
TODO throw example

##### The Tokens Vector
Represented in feature matrix depiction as &psi;

Alphabetically-sorted single-dimension NumPy array containg all tokens for a training corpus with length denoted by &Theta;


example: 
TODO throw an example here


### Method

#### Data Curation
In `src/preprocessor`, the `collect_data(server, observations)` function allows the aggregation of `observations` number of data points by iteratively sending GET and POST requests to the server through the provided sample code functions.

The [[important]] pieces of information that are extracted and saved include: 
- `label`: extracted from `server.ans`, functions as the observation label, enabling us to train our model to produce accurate results
- `blob`: a string storing the hexadecimal representation of the binary blob given by `server.binary`
- `possible_ISAs`: the list of six possible ISAs that the binary blob could have been produced by. In including this feature, I hypothesize that the classifier may recognize at some point that it will never produce a successful classification of an observation when the label is not included in the `possible_ISAs` list


#### Tokenization 
Bag-of-words 

#### Feature Extraction 
#### Classification 
##### Training 
##### Testing

### Misc
- My cookie: 0ccbc0c1-e093-4329-9fea-76f78f2b076c
- [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726)
- 
