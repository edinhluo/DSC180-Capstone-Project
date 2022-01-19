### Required Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_infected(A,y):
    infected_vec = np.zeros((1,A.shape[1]))
    for x in range(A.shape[0]):
        if y[x] == 0:
            infected_vec += A[x]
    return (infected_vec == 0) * 1

def run_sims(n, func):
    
    ### Running n sims of specified Arya Sim for simple adjustments

    results = []

    for x in range(n):
        results.append(func)
        
    ### Determining the Mean of each measurement
    mean = np.mean(results, axis=0).tolist()
    
    category = ["TP","TN","FP","FN","Accuracy"]
    df = pd.DataFrame([mean], columns = category)
    return df


def arya_sim(tests, samples, infected, per_row):
    
    ### Create the Pooling Matrix
    A = np.zeros((tests, samples))
    
    for row in range(tests):
        
        ### Randomly determine which samples belong to a pool 
        random = np.random.choice(samples, size = per_row, replace=False)
        
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
    
    # ### "Reverse Engineering" to determine the accuracy of the algorithm
    # hidden_X = np.zeros((1,samples))
    
    # ### Finds which pools have negative tests
    # for test in range(tests):
    #     if Y[test] == 0:
    #         hidden_X += A[test]
    
    # ### Samples not part of "negative pools" are considered "infected"
    # hidden_infected = (hidden_X == 0) * 1

    hidden_infected = find_infected(A, Y)
    
    ### Determine TP, TN, FP, FN, Accuracy
    TP_ = np.logical_and(hidden_infected, X)
    FP_ = np.logical_and(hidden_infected, np.logical_not(X))
    TN_ = np.logical_and(np.logical_not(hidden_infected), np.logical_not(X))
    FN_ = np.logical_and(np.logical_not(hidden_infected), X)

    TP = sum(TP_[0])
    FP = sum(FP_[0])
    TN = sum(TN_[0])
    FN = sum(FN_[0])
    acc = (TP + TN) / samples
        

    return TP, TN, FP, FN, acc




    ### Some Visulizations of findings
def sims_vis_adjust(n, lst):
    ### n = # of sims
    ### simple adjustment to make sims a bit easier
    
    tb = []
    category = ["Tests","Samples","Infected","Pool size","TP","TN","FP","FN","Accuracy"]
    
    ### lst is a list of the list
    for x in lst:
        tests, samples, infected, per_row = x[0], x[1], x[2], x[3]
        df = run_sims(n, arya_sim(tests, samples, infected, per_row))
       
        ### get results from run_sims
        result = df.to_numpy()[0]
    
        ### append together
        x.extend(result)
        tb.append(x)
        
    df_2 = pd.DataFrame(tb, columns = category)
    return df_2


def visual_df(lower,upper, step, param, default_sim = [20, 100,7,10], default_n = 100):
    '''
    Args:
        upper: max # you want to test
        lower: min # you want to test
        step: step 
        param: which parameter you want to vary
           should be one of these:
           {tests, samples, infected, pool size}
        default_sim: default parameters for the tests
        default_n: number of sims
           
    Returns:
        pd.Dataframe: dataframe with results user
        for later visulization
    '''
    
    ranges = list(range(lower,upper,step))
    
    loc = {'tests':0, 'samples':1,'infected':2,'pool size': 3}
    
    vary_loc = loc[param]
    tasks = []
    for i in ranges:
        hold = default_sim.copy()
        hold[vary_loc] = i
        tasks.append(hold)
    
    tb = sims_vis_adjust(default_n, tasks)
    return tb





def plot_graph(df,X,Y,title = None):
    # X,Y needs to be string
    # X,Y needs to be one of the df columns
    
    fig = plt.figure() 
    default_size = fig.get_size_inches() 
    fig.set_size_inches( (default_size[0]*5, default_size[1]*2) ) 
    plt.rcParams.update({'font.size': 22})
    plt.plot(df[X], df[Y])
    if title is not None:
        plt.title(title)
    plt.xlabel(X)
    plt.ylabel(Y)

def eval_model(predicted_X, true_X):
    TP_ = np.logical_and(predicted_X, true_X)
    FP_ = np.logical_and(predicted_X, np.logical_not(true_X))
    TN_ = np.logical_and(np.logical_not(predicted_X), np.logical_not(true_X))
    FN_ = np.logical_and(np.logical_not(predicted_X), true_X)

    TP = sum(TP_[0])
    FP = sum(FP_[0])
    TN = sum(TN_[0])
    FN = sum(FN_[0])
    acc = (TP + TN) / (TP+FP+TN+FN)
    category = ["TP","TN","FP","FN","Accuracy"]
    df = pd.DataFrame([[TP,TN,FP,FN,acc]], columns = category)
    return df





