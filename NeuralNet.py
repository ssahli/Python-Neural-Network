import numpy as np
import timeit
from sys import stdout

class NeuralNetwork:

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def __init__(self, n_inp, n_hid, n_out, alpha, iter):
        '''
            n_inp:    Neurons in input layer
            n_hid:    Neurons in hidden layer
            n_out:    Neurons in output layer
            W1:       Weights between input and hidden layer
            W2:       Weights between hidden and output layer
            alpha:    Learning rate for backprop
            iter:     Number of iterations (epochs)
        '''
        # Learning rate, learning momentum, iterations
        self.alpha = alpha
        self.iter = iter

        # Number of neurons per layer
        self.num_inputs = n_inp
        self.num_hidden = n_hid
        self.num_output = n_out

        # Initialize sums of products of x's and W's to zero
        self.z1 = np.asarray([0.0] * self.num_hidden)
        self.z2 = np.asarray([0.0] * self.num_output)

        # Initialize deltas to zero
        self.error = np.asarray([0.0] * self.num_output)
        self.d0 = np.asarray([0.0] * self.num_output)
        self.d1 = np.asarray([0.0] * self.num_hidden)

        # Each row is a set of weights from one node of the previous layer
        self.W1 = np.random.randn(self.num_inputs, self.num_hidden)
        self.W2 = np.random.randn(self.num_hidden, self.num_output)
        # Bias weights
        self.b1 = np.random.randn(self.num_hidden)
        self.b2 = np.random.randn(self.num_output)



    def forward(self, x):
        '''
            x: Input data in vector form
            z: Sum of the products of input x and weights W
            a: Activation of layer, sigmoid of z
                a0: input layer
                a1: hidden layer
                a2: output layer

            Recall: a(i) = sigmoid ( a(i-1).W(i) + b(i) )
        '''
        # Set activation of input layer to input data
        self.a0 = np.asarray(x)

        # Calculate activation of hidden layer
        self.z1 = np.dot(self.a0, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Calculate activation of output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2



    def backward(self, t):
        '''
            t: Target output in vector form
            a: Activation of neuron, sigmoid of z
            d: Deltas: how far off the hypothesis is from the target
                d0: Changes for output weights
                d1: Changes for hidden weights
        '''
        # Calculate output layer error
        self.error = t - self.a2
        self.d0 = self.error * self.alpha

        # Calculate hidden layer error
        self.d1 = np.dot(self.d0, self.W2.T)
        self.d1 *= self.dsigmoid(self.a1) * self.alpha

        # Update output layer error
        self.W2 += np.outer(self.a1, self.d0)
        self.b2 += self.d0

        # Update hidden layer error
        self.W1 += np.outer(self.a0, self.d1)
        self.b1 += self.d1



    def train(self, data):
        '''
            Train the neural net on a training set. First split the set into
            training data and their targets. Each feed forward and back prop
            is called per row of data. Entire dataset is run iter times. The
            whole training process is timed, and time spent is printed out.
        '''
        # Split data into data and targets
        train_data = data[:, :self.num_inputs]
        target = data[:, self.num_inputs:]
        m = len(train_data)
        iteration_error = 0
        iteration_errors = np.zeros(shape=(self.iter, 1))

        # Training process
        start = timeit.default_timer()
        for i in range(self.iter):
            for j in range(m):
                self.forward(train_data[j])
                self.backward(target[j])
                iteration_error += abs(sum(self.error))
            iteration_error /= m
            stdout.write("\rEpochs trained: %i" % (i+ 1) + "   Error: %f" % iteration_error)
            stdout.flush()
            iteration_errors[i] = iteration_error
        stop = timeit.default_timer()

        print "\nTime to train: " + str(round((stop - start), 3)) + " seconds"

        return iteration_errors



    def test(self, data):
        '''
            Create a hypothesis for every test input. Always calculates a
            hypothesis, rather than guessing 'none of the above' (for examples
            where every possible outcome from sigmoid is less than 0.5).
        '''
        m = len(data)
        test_data = data[:, :self.num_inputs]
        target = data[:, self.num_inputs:]
        hypothesis = np.zeros(shape=(m, self.num_output))
        mask = np.ones(hypothesis[0].shape, dtype = bool)

        # Feed forward the test set
        print "Testing..."
        for i in range(m):
            hypothesis[i] = self.forward(test_data[i])
            best = hypothesis[i].argmax()
            hypothesis[i,best] = 1.0
            mask[best] = 0.0
            hypothesis[i,mask] = 0.0
            mask[best] = 1.0

        # Calculate percent of correct hypotheses
        correct = 0.0
        for i in range(m):
            if np.array_equal(hypothesis[i], target[i]):
                correct += 1.0 / m

        print "Percent correct: " + str(round(correct, 4)*100) + "%"

        return correct
