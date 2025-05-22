import numpy as numpy
import matplotlib.pyplot as pyplot

from typing import Tuple

# DATA CREATION

#generates random 2d points (X) and labels (Y) (1 if x>y, else 0)
def createDataset(nPoints: int = 100) -> Tuple[numpy.ndarray, numpy.ndarray]:
    numpy.random.seed(0)
    X = numpy.random.rand(nPoints, 2)
    Y = (X[:, 0] > X [:, 1]).astype(int).reshape(-1,1)

    return X, Y 

# INITIALIZATION

#initializes weights and biases an NN with 1 hidden layer
def initParams(inputSize: int, hiddenSize: int, outputSize: int):
    #first set of weights/biases
    W1 = numpy.random.randn(inputSize, hiddenSize)
    b1 = numpy.zeros(1, hiddenSize)
    
    #second set of weights/biases
    W2 - numpy.random.randn(hiddenSize, outputSize)
    b2 = numpy.zeros(1,outputSize)

    return W1, b1, W2, b2

# ACTIVATION
def sigmoid(z: numpy.ndarray) -> numpy.ndarray:
    return 1/(1+numpy.exp(-z))

# def sigmoidDeriv(z:numpy.ndarray) -> nu
