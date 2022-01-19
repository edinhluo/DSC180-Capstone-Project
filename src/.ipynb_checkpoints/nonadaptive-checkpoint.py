import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pool_func

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

def arya_sim(tests, samples, infected, **kwargs):

    pool_size = int(np.round(samples/infected))
    
    A,X,Y = pool_func.pool_sim(tests, samples, infected, pool_size)

    hidden_infected = pool_func.find_infected_binary(A, Y)
    
    df = eval_model(hidden_infected, X)

    return df

### Same results were being printed, ie: a function was not simulated n times, but 1 result replicated n times
def run_nonadaptive(data, **kwargs):
    ##n, tests, samples, infected
    df = pd.read_csv(data)
    os.makedirs(outdir, exist_ok = True)
    means = []
    for index,row in df.iterrows():
        infected = row['Infected']
        samples = row['Population']
        tests = row['Pool Size']
        ### Running n sims of specified Arya Sim for simple adjustments
        results = []
        acc_results = []

        for x in range(100):
            sim = arya_sim(tests, samples, infected).iloc[0]
            results.append(sim)
            acc_results.append(sim[2])
        
        ### Determining the Mean of each measurement
        means_one = np.mean(results, axis=0).tolist()
        means.append(means_one)
    
    category = ["Sens", "Spec", "Accuracy"]
    df = pd.DataFrame(means, columns = category)
    df.to_csv('data/report/nonadaptive.csv', index = False)
    return df

def plot_fig(X,Y):  
    fig = plt.figure() 
    default_size = fig.get_size_inches() 
    fig.set_size_inches( (default_size[0]*5, default_size[1]*2) )    
    plt.rcParams.update({'font.size': 22})
    plt.plot(X, Y)
    plt.ylabel('Accuracy')