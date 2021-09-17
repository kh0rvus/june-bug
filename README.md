june-bug
================

Machine Learning Multi-Class Classifier to Solve Praetorian's MLB Challenge

### The Problem
Server generates a random program and compiles it into a randomly chosen ISA (one of 12)

the instruction aligned binary blob produced by the server is then provided to the agent, and the agent must classify and label the ISA that produced the binary 

### Features of Interest

- The binary is instruction aligned, meaning the features we select will need to consider instruction size, rather than analyzing each hex digit, we want to be able to tell our agent to analyze the instructions
- 

### The Oracle
The post(target) method takes in a guess, and returns the streak, **correct answer**, and hash if challenge was complete. This is important because the correct answer allows us to avoid the problem of manual observation labeling, largely increasing the data we can efficiently equip our model with

### Method

#### Data Curation
In `src/preprocessor`, the `collect_data(server, observations)` function allows the aggregation of `observations` number of data points by iteratively sending GET and POST requests to the server through the provided sample code functions.

The important pieces of information that are extracted and saved include: 
- `label`: extracted from `server.ans`, functions as the observation label, enabling us to train our model to produce accurate results
- `blob`: a string storing the hexadecimal representation of the binary blob given by `server.binary`
- `possible_ISAs`: the list of six possible ISAs that the binary blob could have been produced by. In including this feature, I hypothesize that the classifier may recognize at some point that it will never produce a successful classification of an observation when the label is not included in the `possible_ISAs` list


#### Feature Extraction

#### Classification

##### Training

##### Prediction

### Misc
- My cookie: 0ccbc0c1-e093-4329-9fea-76f78f2b076c




