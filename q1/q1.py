import numpy as np
from random import random, seed

from tensorflow.examples.tutorials.mnist import input_data

#load mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def initialise_weights_biases():
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
    print(1 / (1 + np.exp(np.negative(array))))
    return 1 / (1 + np.exp(np.negative(array)))

def relu_activ(array):
    return np.maximum(array, 0)

def softmax_activ(array):
    logits = np.exp(array)
    return logits / np.sum(logits, axis = 1, keepdims = True)

def cross_entropy_loss(probability_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted = probability_array[np.arange(len(probability_array)), indices]
    log_predicted = np.log(predicted)
    loss = -1.0 * np.sum(log_predicted) / len(log_predicted)
    return loss

sigmoid_activ([1,2,3,4])

# def l2_regularisation(lambda_val, w1, w2):
#     w1_loss = 0.5 * lambda_val * np.sum(w1 * w1)
#     w2_loss = 0.5 * lambda_val * np.sum(w2 * w2)
#     return w1_loss + w2_loss
