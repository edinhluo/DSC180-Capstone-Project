from multiprocessing import pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Generates NonAdaptive Pooling
def generate_pool(tests, samples, pool_size = None):

    if pool_size is None:
        pool_size = 32

    ### Create the Pooling Matrix
    A = np.zeros((tests, samples))

    for row in range(tests-5):
        
        ### Randomly determine which samples belong to a pool 
        random = np.random.choice(samples, size = pool_size, replace=False)
        
        for sample in random:
            A[row][sample] = 1

    ### Determine which samples have not been tested
    remaining = (A.sum(axis = 0) == 0).nonzero()[0]
    size = len(remaining)

    ### Test the remaining Samples
    for row in range(5):

        while size < pool_size:
            random = np.random.choice(samples, size = pool_size, replace=False)
            random[0:size] = remaining
            remaining = np.unique(random)
            size = len(remaining)

        random = np.random.choice(remaining, size = pool_size, replace=False)
        
        for sample in random:
            A[row][sample] = 1
        
        remaining = (A.sum(axis = 0) == 0).nonzero()[0]
        size = len(remaining)

    ### Output Pooling Matrix
    return A

### Generate Infected Samples Vector
def generate_infected(samples, infected):

    ### Create the "Infected" Binary X Vector
    X = np.zeros((1, samples))
    
    ### Randomly determine the infected samples
    random = np.random.choice(samples, size = infected, replace = False)
    
    for positive in random:
        X[0][positive] = 1

    return X

### Determine the Output Pooled Tests
def eval_matrix(A,X):
    Y = ((np.matmul(A, X.T) > 0)*1)
    return Y

### Generates Testing Simulation
def pool_sim(tests, samples, infected, pool_size=None):
    
    A = generate_pool(tests, samples, pool_size)

    X = generate_infected(samples, infected)
        
    Y = eval_matrix(A,X)

    return A, X, Y

### Returns Pools with Positive Result
def find_positive_pools(A,Y):
    posY = Y.nonzero()[0]
    pools = []
    for pool in posY:
        positive = (A[pool].nonzero()[0])
        pools.append(positive)
    return pools

### Outputs Binary Vector indicating Positive Samples
### If Index set to True, output positive samples index numbers
def find_infected_binary(A,y, index = None):
    infected_vec = np.zeros((1,A.shape[1]))
    for x in range(A.shape[0]):
        if y[x] == 0:
            infected_vec += A[x]

    infected_vec = (infected_vec == 0) * 1
    
    if index is True:
        return infected_vec[0].nonzero()[0].tolist()

    return infected_vec

### --------------------------------------------------------------- ###

### Create Disjoint Pooling Matrix where samples tested evenly
def create_disjoint(tests, samples, infected, pool_size = None):
    
    ### Create the Pooling Matrix
    A = np.zeros((tests, samples))

    if pool_size is None:
        pool_size = 32
        
    start = 0

    ### Shuffles samples prior to assigning pools
    random = np.random.choice(samples, size = samples, replace=False)

    for row in range(tests):
        
        end = start + pool_size

        if end > samples:
            end = samples

        subset = random[start:end]

        for sample in subset:
            A[row][sample] = 1

        start += pool_size

        if end == samples:
            start = 0
            random = np.random.choice(samples, size = samples, replace=False)

            if (tests - row) <= infected:
                remainder = tests - row - 1
                if remainder:
                    pool_size = int(np.ceil(samples/remainder))

    return A

### Creates Disjoint Simulation
def disjoint_pool_sim(tests, samples, infected, pool_size = None):

    A = create_disjoint(tests,samples, infected, pool_size)
    
    ### Create the "Infected" Binary X Vector
    X = generate_infected(samples, infected)
        
    ### Find the resulting Y Vector, outputting a binary result    
    Y = eval_matrix(A,X)

    return A, X, Y

### --------------------------------------------------------------- ###

### Returns TP,TN,ACC
def eval_model(predicted_X, true_X):
    TP_ = np.logical_and(predicted_X, true_X)
    FP_ = np.logical_and(predicted_X, np.logical_not(true_X))
    TN_ = np.logical_and(np.logical_not(predicted_X), np.logical_not(true_X))
    FN_ = np.logical_and(np.logical_not(predicted_X), true_X)

    TP = sum(TP_[0])
    FP = sum(FP_[0])
    TN = sum(TN_[0])
    FN = sum(FN_[0])

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    acc = (TP + TN) / (TP+FP+TN+FN)
    category = ["Sens", "Spec", "Accuracy"]
    df = pd.DataFrame([[TPR, TNR, acc]], columns = category)
    return df

def run_nonadaptive(tests, samples, infected, pool_size, simulations):
    acc = []
    for _ in range(simulations):
        A,X,Y = pool_sim(tests,samples, infected, pool_size)
        predX = find_infected_binary(A,Y)
        acc.append(eval_model(predX,X)["Accuracy"][0])
    return acc

def run_disjoint(tests, samples, infected, pool_size, simulations):
    acc = []
    for _ in range(simulations):
        A,X,Y = create_disjoint(tests,samples, infected, pool_size)
        predX = find_infected_binary(A,Y)
        acc.append(eval_model(predX,X)["Accuracy"][0])
    return acc

def NA_results(tests, samples, infected, pool_size, sims):
    NA_runs = run_nonadaptive(80,1000, 10, 100 ,1000)
    NA_runs = run_nonadaptive(tests, samples, infected, pool_size, sims)

    NA_str1 = "For a population size of " + str(samples) + ", " +str(tests) + " tests, " + str(infected) + " infected, pool size " + str(pool_size) +" in " + str(sims) + " simulations of the Non Adaptive Scheme:"
    NA_str2 = "Average accuracy:" + str(NA_runs[0]) 
    NA_str3 = "Average standard deviation:" +str(NA_runs[1])
    NA_str4 = "Over " + str(NA_runs[0]*samples - infected) + " people were detected COVID-19 negative, while the remaining are declared positive and tested further." 
    
    print(NA_str1)
    print(NA_str2)
    print(NA_str3)
    print(NA_str4)

def DJ_results(tests, samples, infected, pool_size, sims):
    DJ_runs = run_disjoint(40,500,5,32,1000)
    DJ_runs = run_disjoint(tests, samples, infected, pool_size, sims)

    DJ_str1 = "For a population size of " + str(samples) + ", " +str(tests) + " tests, " + str(infected) + " infected, pool size " + str(pool_size) +" in " + str(sims) + " simulations of the Disjoint Scheme:"
    DJ_str2 = "Average accuracy:" + str(DJ_runs[0]) 
    DJ_str3 = "Average standard deviation:" +str(DJ_runs[1])
    DJ_str4 = "Over " + str(DJ_runs[0]*samples - infected) + " people were detected COVID-19 negative, while the remaining are declared positive and tested further." 
    
    print(DJ_str1)
    print(DJ_str2)
    print(DJ_str3)
    print(DJ_str4)





    