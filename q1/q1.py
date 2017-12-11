import numpy as np
from random import random, seed

from tensorflow.examples.tutorials.mnist import input_data

#load mnist data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

nb_epochs = 10

layer_1_weights = np.random.normal(0, 1, [784, 100])
layer_1_biases = np.zeros((1, 100))

layer_2_weights = np.random.normal(0, 1, [100, 50])
layer_2_biases = np.zeros((1,50))

output_layer_weights = np.random.normal(0, 1, [50, 10])
output_layer_biases = np.zeros((1, 10))

# TODO do we need to randomly initialise biases too?
# TODO try randomly intialise using code below:
# hidden_layer_1 = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(layer_1_n_nodes)]

def sigmoid_activ(array):
    return 1 / (1 + np.exp(np.negative(array)))

def sigmoid_deriv(array):
    return np.multiply(array, np.subtract(1.0, array))

def relu_activ(array):
    return np.maximum(array, 0)

def relu_deriv(array):
    return np.greater(array, 0).astype(int)

def softmax_activ(array):
    logits = np.exp(array)
    return logits / np.sum(logits, axis = 1, keepdims = True)

def cross_entropy_loss(probability_array, labels):
    indices = np.argmax(labels, axis = 1).astype(int)
    predicted = probability_array[np.arange(len(probability_array)), indices]
    loss = -1.0 * np.sum(np.log(predicted)) / len(np.log(predicted))
    return loss

def train():
    for step in range(nb_epochs):
        input_layer = np.dot(mnist.train.images, layer_1_weights)
        hidden_layer_1 = sigmoid_activ(input_layer + layer_1_biases)
        hidden_layer_2 = relu_activ(np.dot(hidden_layer_1, layer_2_weights) + layer_2_biases)
        output_layer = np.dot(hidden_layer_2, output_layer_weights) + output_layer_biases
        output_probs = softmax_activ(output_layer)

    loss = cross_entropy_loss(output_probs, mnist.train.labels)
    print(loss)

    # TODO backpropagate

    # output_error_signal = (output_probs - mnist.train.labels) / output_probs.shape[0]

relu_deriv([-1, 1, 2, 3, 4])
