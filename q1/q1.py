import numpy as np
from random import random

from tensorflow.examples.tutorials.mnist import input_data

#load mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

nb_epochs = 20
loss_array = []

def sigmoid_activ(x):
    return 1.0 / (1.0 + np.exp(np.negative(x)))

def sigmoid_deriv(x):
    return np.multiply(sigmoid_activ(x), np.subtract(1.0, sigmoid_activ(x)))

def relu_activ(x):
    return np.maximum(x, 0)

def relu_deriv(x):
    return np.greater(x, 0).astype(int)

def softmax_activ(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis = 1, keepdims = True)

def cross_entropy(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def train():
    learning_rate = .1

    # initialise weights and biases
    layer_1_weights = np.random.normal(0, 1, [784, 100])
    layer_1_biases = np.zeros((1, 100))

    layer_2_weights = np.random.normal(0, 1, [100, 50])
    layer_2_biases = np.zeros((1,50))

    output_layer_weights = np.random.normal(0, 1, [50, 10])
    output_layer_biases = np.zeros((1, 10))

    for step in range(nb_epochs):
        input_layer = np.dot(mnist.train.images, layer_1_weights)

        hidden_1_weighted_sum = np.add(np.dot(mnist.train.images, layer_1_weights), layer_1_biases)
        hidden_layer_1 = sigmoid_activ(hidden_1_weighted_sum)

        hidden_2_weighted_sum = np.add(np.dot(hidden_layer_1, layer_2_weights), layer_2_biases)
        hidden_layer_2 = relu_activ(hidden_2_weighted_sum)

        output_layer_weighted_sum = np.add(np.dot(hidden_layer_2, output_layer_weights), output_layer_biases)
        output_layer = softmax_activ(output_layer_weighted_sum)

        # cross entropy loss
        loss = np.mean(cross_entropy(output_layer, mnist.train.labels))
        loss_array.append(loss)

        # back propagation starts
        out_error = (output_layer - mnist.train.labels) / output_layer.shape[0]

        grad_output_weights = np.dot(hidden_layer_2.T, out_error)
        grad_output_biases = np.sum(out_error, axis = 0, keepdims = True)

        hidden_2_delta = np.dot(out_error, output_layer_weights.T)
        hidden_2_delta = np.multiply(hidden_2_delta, relu_deriv(hidden_2_weighted_sum))
        grad_hidden_2_weights = np.dot(hidden_1_weighted_sum.T, hidden_2_delta)
        grad_hidden_2_biases = np.sum(hidden_2_delta, axis = 0, keepdims = True)

        hidden_1_delta = np.dot(hidden_2_delta, layer_2_weights.T)
        hidden_1_delta = np.multiply(hidden_1_delta, sigmoid_deriv(hidden_1_weighted_sum))
        grad_hidden_1_weights = np.dot(mnist.train.images.T, hidden_1_delta)
        grad_hidden_1_biases = np.sum(hidden_1_delta, axis = 0, keepdims = True)

        output_layer_weights -= learning_rate * grad_output_weights
        output_layer_biases -= learning_rate * grad_output_biases
        layer_2_weights -= learning_rate * grad_hidden_2_weights
        layer_2_biases -= learning_rate * grad_hidden_2_biases
        layer_1_weights -= learning_rate * grad_hidden_1_weights
        layer_1_biases -= learning_rate * grad_hidden_1_biases

        print('loss at step ', step, ": ", np.mean(loss))

train()

#keras model
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
h1 = model.add(Dense(100, activation='sigmoid', input_shape=(mnist.train.images.shape[1],)))
h2 = model.add(Dense(50, activation='relu'))
out = model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

H = model.fit(x=mnist.train.images, y=mnist.train.labels,
            batch_size=128,
            epochs=nb_epochs)

keras_loss_array = H.history['loss']

# plot graph
import plotly
import plotly.graph_objs as go

trace0 = go.Scatter(
    y = loss_array,
    x = list(range(1,20)),
    mode = 'lines'
)

trace1 = go.Scatter(
    y = keras_loss_array,
    x = list(range(1,20)),
    mode = 'lines'
)

plot_data = [trace0, trace1]
layout = go.Layout(
    xaxis=dict(
        title="Epoch Number"
    ),
    yaxis=dict(
        title='Loss'
    )
)

fig = go.Figure(data=plot_data, layout=layout)
plotly.offline.plot(fig, filename='loss-graph')
