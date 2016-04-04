import numpy as np

# Load data. 1797 x 65
data = np.loadtxt("digits.csv", delimiter = ',')

# New variable for data destination. Add 9 new columns to the dataset, init to 0
transform = np.zeros((data.shape[0],data.shape[1]+9))
transform[:,:-9] = data

# One hot encoding for the response. 10 digits, 10 bits.
for i in range(len(transform)):
    if transform[i,64] == 0:
        transform[i,64] = 1
    else:
        if transform[i,64] == 1:
            transform[i,65] = 1
        elif transform[i,64] == 2:
            transform[i,66] = 1
        elif transform[i,64] == 3:
            transform[i,67] = 1
        elif transform[i,64] == 4:
            transform[i,68] = 1
        elif transform[i,64] == 5:
            transform[i,69] = 1
        elif transform[i,64] == 6:
            transform[i,70] = 1
        elif transform[i,64] == 7:
            transform[i,71] = 1
        elif transform[i,64] == 8:
            transform[i,72] = 1
        elif transform[i,64] == 9:
            transform[i,73] = 1
        transform[i,64] = 0

# Scale the data. Max value is 16.0, which will be the new 1.0
transform[:, :-10] = transform[:, :-10] / float(16.0)

# Write the new data to file
np.savetxt("transformed.csv", transform, delimiter = ',')
