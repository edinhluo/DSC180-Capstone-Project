import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import math
import networkc as nw
from collections import defaultdict
import os


def find_infected(A,y):
    infected_vec = np.zeros((1,A.shape[1]))
    for x in range(A.shape[0]):
        if y[x] == 0:
            infected_vec += A[x]
    return (infected_vec == 0) * 1


def sort_G(G):
    """
    sort nodes in G based on their degree centrality
    return a list of sets with node number and their degree centrality
    """
    
    measure = nx.degree_centrality(G)

    #sort the result diction
    sort = sorted(measure.items(), key=lambda x: x[1])
    sort.reverse()
    return sort

def pool(G,tests, per_row,infected = None):
    ### tests = # of rows
    ### per_row = number of 1 in a row (ppl in pool)
    ###if infected != None, not first round
    
    #if isinstance(G, list):
     #   G = nw.disease_spread(G[0],G[1],G[2])

    samples = len(G)
    track = []
    
    ### Create the matrix
    A = np.zeros((tests, samples))
    
    if infected is not None: 
        nonzero = infected.nonzero()[0]
        pos = list(nonzero)
        #print(len(nonzero))
        #left node
        sort_nx = sort_G(G)
        sorted_nx = [x for x in sort_nx if x[0] in pos]
        
    else:
    
        ### sorted centryality
        sorted_nx = sort_G(G)
    
    loc = [x[0] for x in sorted_nx] #node indx
    
    #last 25% has 70%,50 to 75 quantile has 20% and the rest has 10%
    mid = int(len(loc) * 0.5)
    q75 = int(len(loc) * 0.75)
    q25 = int(len(loc) * 0.25)
    
    if (mid == 0) | (q75 == 0) | (q25 == 0) :
        #do individual testing
        if len(loc) == 0:return A,track
        track.extend(loc[0])
       
        for sample in track:
            A[0][sample] = 1
        return A,track
    
    ptop = 0.10 / mid
    p5075 = 0.20 / (q75 - mid)
    plast = 0.70 / (len(loc) - q75)
    
    #generate prob list
    lst = [mid,(q75-mid),(len(loc) - q75)]
    plst = [ptop,p5075,plast]
    prob = []
    for x in range(len(lst)):
        p = [plst[x]] * lst[x]
        prob.extend(p)
    
    for row in range(tests):
        
        ### Randomly determine which samples belong to a pool 
        if (len(loc)<per_row): per_row = 1 #do individual testing
        rd = np.random.choice(loc, p = prob , replace = False, size = per_row)
        #rd = loc[(-1)*per_row:]
        track.extend(rd)
        
        for sample in rd:
            A[row][sample] = 1
    
    #print(track)
   

    return A,track


def find_infected(A,X,infected_vec,track,one):
    if len(track) == 0:return infected_vec,one ###stop
         
    ### Find the resulting Y Vector, outputting a binary result    
    Y = ((np.matmul(A, X.T) > 0)*1)
    #print(Y)
    
    
    for x in range(A.shape[0]):
        if Y[x] == 0:    
            for pos in track:
                infected_vec[pos] = 0
        else:
            if len(track) == 1:
                for t in range(3):
                    one.extend(track)
            one.extend(track)
            
            
    return infected_vec,one


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


def test(data, **kwargs):

    df = pd.read_csv(data)
    #os.makedirs("outdir", exist_ok = True)

    for index,row in df.iterrows():
        n1 = row['network1']
        n2 = row['network2']
        n3 = row['network3']
        tt = row['tests']
        pr = row['per_row']


    G = nw.disease_spread(n1,n2,n3)
    tests = tt
    per_row = pr 

    ### Create the "Infected" Binary X Vector
    X = np.zeros((1, len(G)))
    ### find infeced location
    positive = nw.all_pos(G)
    
    one = [] #track ones
    back = set()
    
    infected = np.ones(len(G))
    
    ### set pos based on network 
    for pos in positive:
        X[0][pos] = 1
        
    for i in range(tests):
        if i == 0:
            A,track = pool(G,1, per_row)
            infected,one = find_infected(A,X,infected,track,one)
            #print(len(infected.nonzero()[0]))
        else:
            A,track = pool(G,1, per_row,infected)
            infected,one = find_infected(A,X,infected,track,one)
            #print(len(infected.nonzero()[0]))
            
            #check time
            infec = check_times(one)
            back.update(infec)
            
            #set them to 0
            for p in infec:
                infected[p] = 0
            
    #put 1s back
    for b in back:
        infected[b] = 1
#     print(len(back))
#     print(len(infected.nonzero()[0]))
    result = eval_model(infected,X)
            
    #print(infected,X)
    result.to_csv('data/report/degreePoolresult.csv', index = False)
    return result



def check_times(lst):
    ### if in one gorup showed up 3 times, must be one
    check = defaultdict(int)
    inf = []
    for i in lst:
        check[i] += 1
    for x in check.keys():
        if check[x] >= 5:
            inf.append(x)
    return inf


