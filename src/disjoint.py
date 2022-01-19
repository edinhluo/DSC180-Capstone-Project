import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pool_func as pfunc
import nonadaptive as nfunc

def disjoint_pool(max_tests, samples, infected, pool_size = None):

    if pool_size is None:
        pool_size = int(np.ceil(samples/infected))

        if max_tests < infected:
            pool_size = int(np.ceil(samples/max_tests))

    ### Create the Pooling Matrix
    A = np.zeros((max_tests, samples))

    start = 0

    ### Randomly determine which samples belong to a pool
    random = np.random.choice(samples, size = samples,replace=False)
    
    for row in range(max_tests):
        
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

            if (max_tests - row) <= infected:
                remainder = max_tests - row - 1
                if remainder:
                    pool_size = int(np.ceil(samples/remainder))
        
    
    ### Create the "Infected" Binary X Vector
    X = np.zeros((1, samples))
    
    ### Randomly determine the infeced samples
    random = np.random.choice(samples, size = infected, replace = False)
    
    for positive in random:
        X[0][positive] = 1
        
    ### Find the resulting Y Vector, outputting a binary result    
    Y = ((np.matmul(A, X.T) > 0)*1)

    return A, X, Y

def adaptive_disjoint(tests, samples, X, pos_ind):

    pool_size = int(np.ceil(len(pos_ind)/tests))

    ### Create the Pooling Matrix
    A = np.zeros((tests, samples))

    start = 0

    ### Randomly determine which samples belong to a pool
    random = np.random.choice(pos_ind, size = len(pos_ind),replace=False)
    
    for row in range(tests):
        
        end = start + pool_size

        if end > samples:
            end = samples

        subset = random[start:end]

        for sample in subset:
            A[row][sample] = 1

        start += pool_size
        
    ### Find the resulting Y Vector, outputting a binary result    
    Y = ((np.matmul(A, X.T) > 0)*1)

    return A, Y, tests

def equal_sim(tests, samples, infected):
    
    A,X,Y = disjoint_pool(tests, samples, infected)
    hidden_infected = pfunc.find_infected_binary(A,Y)
    df = nfunc.eval_model(hidden_infected, X)
    df["Tests"] = tests
    return df, hidden_infected

def run_equal(data, **kwargs):
    os.makedirs("outdir", exist_ok = True)
    df = pd.read_csv(data)
    means = []
    for index,row in df.iterrows():
        infected = row['Infected']
        samples = row['Population']
        tests = row['Pool Size']
        results = []
        acc_results = []

        for x in range(100):
            sim = equal_sim(tests, samples, infected)[0].iloc[0]
            results.append(sim)
            acc_results.append(sim[2])
        
        ### Determining the Mean of each measurement
        mean_one = np.mean(results, axis=0).tolist()
        means.append(mean_one)
    category = ["Sens", "Spec", "Accuracy", "Tests"]
    df = pd.DataFrame(means, columns = category)
    df.to_csv('data/report/equal.csv',index = False)
    return df

def recursive_pooling(max_tests, samples, infected):

    A,X,Y = disjoint_pool(infected, samples, infected)

    pos_pools = pfunc.find_positive_pools(A,Y)
    tests = infected
    num_pools = len(pos_pools)
    max_infected = infected - num_pools + 1
    confirmed = []

    while (pos_pools) and (tests < max_tests):

        sim_pool = pos_pools[0]

        if len(sim_pool) == 1:
            infected -= 1
            confirmed.append(sim_pool[0])

        elif (max_infected) == 1:

            min_test = int(np.ceil(np.log(len(sim_pool)))+1)
            
            A1, Y1, test = adaptive_disjoint(min_test, samples, X, sim_pool)

            ### Find Infected Pools, outputs list of pools
            infected_pools = pfunc.find_positive_pools(A1,Y1)
            pos_pools += infected_pools
            tests += test
            A = np.vstack((A, A1))

        else:
            A1, Y1, test = adaptive_disjoint(max_infected, samples, X, sim_pool)

            ### Find Infected Pools, outputs list of pools
            infected_pools = pfunc.find_positive_pools(A1,Y1)
            pos_pools += infected_pools
            tests += test
            A = np.vstack((A, A1))

        pos_pools = pos_pools[1:]
        num_pools -= 1
        
        if (num_pools) == 0:
            num_pools = len(pos_pools)
            max_infected = infected - num_pools + 1

            if max_infected < 1:
                print(max_infected)
                break


    for leftover in pos_pools:
        for each in leftover:
            confirmed.append(each)

        
    ### Find the resulting Y Vector, outputting a binary result    
    Y = ((np.matmul(A, X.T) > 0)*1)

    return confirmed, tests