# NN_backpropagating_errors
A Neural Network generated using the backpropagating errors algorithm able to predict the first resonance frequency of a beam

# Introduction and motivation
This project has been created to understand more deeply the backpropagating errors algorithm formulated by Goeffrey Hinton in his famous 1986 paper "Learning representation by back-propagating errors" for the creation of a Neural Network.

The model has been trained to predict the first resonance frequency of an Euler-Bernoulli beam, for which an analytical model is available.
This allows to easily judge the quality of the generated neural network model.

Most of the project can be readapted to build a neural network to predict other physical phenomena.

# Inputs
In train.py:
- epochs iterations for training
- learning rate
- number of training samples
- range of values of training parameters
- number of nodes per each layer

In predict.py
- beam geometry
- beam material

# Outputs
- beam_model.pkl: contains the trained NN model
- weights.txt: contains the tensors containing the final weights of each node
- biases.txt: contains the final biases of each node

# How to use
1. Run train.py
2. Run predict.py

# Note
Relu activation function is used.
