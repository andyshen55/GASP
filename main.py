import numpy as np
import os

CSV_PATH = os.getcwd()  #replace with path to native CSV file
ALPHA = 0.001           #step size for descent
REPS = 10000            #learning iterations 
ENTRIES = 500           #applicants ie. data points
COLS = 7                #relevant categories

#imports and formats dataset
data = np.genfromtxt(CSV_PATH, delimiter = ",")
data = data[1:, 1:]

#expected chances of admission
expected = data[:, COLS].reshape(ENTRIES, 1)

#initialize cost and weight matrices
cost = np.zeros((ENTRIES, COLS + 1))
weights = np.ones((COLS + 1, 1))

#normalizing name column
for i in range(0, ENTRIES):
    cost[i, 0] = 1

#initialize cost matrix with dataset
for i in range(1, COLS + 1):
    for j in range (0, ENTRIES):
        cost[j, i] = expected[j, i - 1]

#matrix of partial derivatives for computing direction of greatest descent
#bias term is stored in data matrix
partials = np.matmul(cost.transpose(), np.matmul(cost, weights) - data)

for steps in range(REPS):
    #update weights ie. move alpha times direction of partials
    weights = weights - (ALPHA * partials)
    #update direction
    partials = np.matmul(cost.transpose(), (np.matmul(cost, weights) - data))
