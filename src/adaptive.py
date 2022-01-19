import pool_func as pfunc
import nonadaptive as nfunc  
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def two_sim(test1, test2, samples, infected):

    pool_size = int(np.round(samples/infected))
    A, X, Y = pfunc.pool_sim(test1, samples, infected, pool_size)
    new_X = sum(pfunc.find_infected_binary(A, Y)[0])

    pool_size = int(np.round(new_X/infected))
    A, X, Y = pfunc.pool_sim(test2, new_X, infected, pool_size)
    pos = sum(pfunc.find_infected_binary(A,Y)[0])

    TP = infected
    FP = pos - infected
    TN = samples - pos
    FN = 0

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    acc = (TP + TN) / (TP+FP+TN+FN)
    category = ["Sens", "Spec", "Accuracy"]
    df = pd.DataFrame([[TPR, TNR, acc]], columns = category)
    return df

def run_adaptive(data, ** kwargs):
    ## n, test1, test2, samples, infected
    df = pd.read_csv(data)
    os.makedirs("outdir", exist_ok = True)
    means = []
    for index,row in df.iterrows():
        test1 = row['first_test']
        test2 = row['second_test']
        samples = row['Population']
        infected = row['Infected']
        acc_results = []
        results = []

        for _ in range(100):
            sim = two_sim(test1, test2, samples, infected).iloc[0]
            results.append(sim)
            acc_results.append(sim[2])

        means_one = np.mean(results, axis=0).tolist() 
        means.append(means_one)
        
    category = ["Sens", "Spec", "Accuracy"]
    df = pd.DataFrame(means, columns = category)
    df.to_csv('data/report/adaptive.csv',index = False)
    return df

def vincent_code(test1, test2, samples, infected):
    pool_size = int(np.round(samples/infected))

    A,X,Y = pfunc.pool_sim(test1, samples, infected, pool_size)
    pos_X = pfunc.find_infected_binary(A, Y)[0]
    indexes_infected = pos_X.nonzero()[1]