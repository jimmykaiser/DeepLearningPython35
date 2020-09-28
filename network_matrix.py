# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially.
        
        eta = learning rate
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            mini_batches_x, mini_batches_y = [], []
            for batch in mini_batches:
                mini_batches_x.append(np.column_stack(tuple([batch[k][0]
                    for k in range(mini_batch_size)])))
                mini_batches_y.append(np.column_stack(tuple([batch[k][1]
                    for k in range(mini_batch_size)])))
            i = 0
            for x, y in zip(mini_batches_x, mini_batches_y):
                self.update_mini_batch(x, y, eta, mini_batch_size, i)
                i = i + 1
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, x, y, eta, mini_batch_size, i):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        # for x, y in mini_batch:
        # Find partial derivatives of cost with respect to biases and weights
        nabla_b, nabla_w = self.backprop(x, y, i)
        # Add up bias and weight gradients in mini batch
        nabla_b = [nb.sum(axis=1) for nb in nabla_b]
        # nabla_b = [np.asmatrix(nb.sum(axis=1)).transpose() for nb in nabla_b]
        # nabla_w = [nw.sum(axis=0) for nw in delta_nabla_w]
        # Update weights and biases in network
        # by subtracting the gradient for mini batch (go downhill)
        self.weights = [w-(eta/mini_batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mini_batch_size)*nb.reshape((nb.shape[0], 1))
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, i):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        mini_batch_size = x.shape[1]
        nabla_b = [np.zeros((mini_batch_size, b.shape[0])) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        ## Feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # go through network and find dot product of weights and x at each layer
            # then add bias and apply sigmoid
            B = np.tile(b, (1, mini_batch_size))
            z = np.dot(w, activation)+B
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        ## Backward Pass
        # BP1: compute gradient of error of output layer
        # one gradient per neuron (10 total)
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book. Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on. It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # BP2: compute gradient of error of this layer for each neuron
            # in terms of error of next layer
            # works because output of next layer is a function 
            # of output of this layer
            # for first neuron in layer:
            # np.dot(self.weights[-l+1].transpose()[0,:], delta[:,0]) * sp[0]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # BP3: compute cost gradient with respect to biases
            # one for each neuron in layer (30, 1)
            nabla_b[-l] = delta
            # BP4: compute cost gradient with respect to weights
            # based on error gradient of this layer and activations
            # of previous layer
            # one for each neuron+weight in layer (30, 784)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x 
        partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
