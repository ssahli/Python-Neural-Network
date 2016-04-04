import matplotlib.pyplot as plt
import pylab
import numpy as np
from sklearn import datasets
from NeuralNet import NeuralNetwork

# Tunable parameters. Go nuts.
HIDDEN_NODES = 128
LEARNING_RATE = 0.3
ITERATIONS = 50
VIEW_EXAMPLES = True
VIEW_PLOT = True


'''
    Lets view a few examples from the original dataset.
    source of code: scikit-learn.org
'''
if VIEW_EXAMPLES == True:
    digits = datasets.load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(4, 3, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)
    pylab.show()



'''
    Load the data. For this demo, we're using sklearn's digits dataset
    Digits are 8x8 pixel images. Each row is one image, in a linear format,
    where columns 65-74 correspond to one hot encoded responses representing
    digits 0 through 9. 1797 rows 74 columns
'''
data = np.loadtxt("transformed.csv", delimiter = ',')
m = len(data)

# Split the data into training set and test set.
train_set = data[:(3*m/4),:]
test_set = data[m/4:,:]

# Instantiate a new neural network. 64 input, 64 hidden, 10 output nodes.
NN = NeuralNetwork(64,HIDDEN_NODES,10,LEARNING_RATE,ITERATIONS)

# Train on the training set, test on the test set. The test() function
# will print out the percent correctness on the test set.
errors = NN.train(train_set)
NN.test(test_set)



# Plot the error curve
if VIEW_PLOT == True:
    plt.plot(errors)
    plt.title("Average Error Per Iteration On Training Set")
    plt.xlabel("Iteration")
    plt.ylabel("Average Error")
    pylab.show()
