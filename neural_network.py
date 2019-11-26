import sys
import random
import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_dimensions):
        # the number of the layers in NeuralNetwork
        self.no_of_layers = len(layer_dimensions)
        self.layer_dimensions = layer_dimensions
        # Custom weight initialization (course 4, Neural Networks) with mu=0 and std = 1/sqrt(number of connections for that neuron)
        self.biases = [np.random.randn(y, 1) for y in layer_dimensions[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(layer_dimensions[:-1], layer_dimensions[1:])]

    def feedforward(self, a):
        """This function is only used to get activations for evaluation.

        It does NOT use dropout"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def fit(self, training_data, epochs, eta=0.5, mini_batch_size=1):
        no_of_mini_batches = len(training_data)//mini_batch_size
        if len(training_data) % mini_batch_size > 0:
            no_of_mini_batches += 1

        for j in range(epochs):
            # shuffle
            random.shuffle(training_data)
            for i in range(0, no_of_mini_batches):
                mini_batch = training_data[i *
                                           mini_batch_size: (i + 1)*mini_batch_size]
                self.update_mini_batch(mini_batch, eta, len(training_data))
            # self.update(training_data, eta, regularization_parameter)

            # testing accuracy
            nailed_cases = self.get_nailed_cases(training_data)
            print(f"Epoch {j}: {nailed_cases}/{len(training_data)}\n")

    def get_predictions(self, X):
        return [np.argmax(self.feedforward(x)) for x in X]

    def update_mini_batch(self, mini_batch, eta, training_data_length):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # L2 regularization
        self.weights = [w + vw for w, vw in zip(self.weights, nabla_w)]
        self.biases = [b + vb for b, vb in zip(self.biases, nabla_b)]

    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Using softmax for the last layer
        # Course 4, slide 27 (Neural Networks)
        # sum_of_values = sum([np.exp(z) for z in zs[-1]])
        # activations[-1] = [np.exp(z)/sum_of_values for z in zs[-1]]

        # # Using the derivative of Cross Entropy here
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.no_of_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def get_nailed_cases(self, test_data):
        """Returns how many cases it nailed from the test_data"""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
