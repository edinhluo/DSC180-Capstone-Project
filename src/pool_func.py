import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pool_sim(tests, samples, infected, pool_size):
    
    ### Create the Pooling Matrix
    A = np.zeros((tests, samples))
    
    for row in range(tests):
        
        ### Randomly determine which samples belong to a pool 
        random = np.random.choice(samples, size = pool_size, replace=False)
        
        for sample in random:
            A[row][sample] = 1
    
    ### Create the "Infected" Binary X Vector
    X = np.zeros((1, samples))
    
    ### Randomly determine the infeced samples
    random = np.random.choice(samples, size = infected, replace = False)
    
    for positive in random:
        X[0][positive] = 1
        
    ### Find the resulting Y Vector, outputting a binary result    
    Y = ((np.matmul(A, X.T) > 0)*1)

    return A, X, Y

def find_positive_pools(A,Y):
    posY = Y.nonzero()[0]
    pools = []
    for pool in posY:
        positive = (A[pool].nonzero()[0])
        pools.append(positive)
    return pools

### Finds who is in an infected pool
def find_infected_binary(A,y):
    infected_vec = np.zeros((1,A.shape[1]))
    for x in range(A.shape[0]):
        if y[x] == 0:
            infected_vec += A[x]
    return (infected_vec == 0) * 1

### Finds how many times a person is in an infected pool
def find_infected_counted(A,y):
    infected_vec = np.zeros((1,A.shape[1]))
    for x in range(A.shape[0]):
        if y[x]:
            infected_vec += A[x]
    return infected_vec