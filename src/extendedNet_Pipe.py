from seirsplus.models import *
from seirsplus.networks import *
from seirsplus.sim_loops import *
from seirsplus.utilities import *
import networkx 
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os


def find_nodeState_extend(G,model):
    """
    input: extended model's network (model.G)  model
    return a dict with each states' nodes

    using format like:  to get particular categoy node
    find_nodeState_extend(model)['I_sym']
    """
    state = {1:'S',2:'E',3:'I_pre',4:'I_sym',5:'I_asym',6:'H',7:'R',8:'F',11:'Q_S',12:'Q_E',13:'Q_pre',
            14:'Q_sym',15:'Q_asym',17:'Q_R'}
    category = ['S','E','I_pre','I_sym','I_asym','H','R','F','Q_S','Q_E','Q_pre','Q_sym','Q_asym','Q_R']
    
    dic = {k:[] for k in category}
    for x in range(len(G)):
        val = model.X[x][0]
        cat = state[val]
        dic[cat].append(x)
        
    return dic


def posNode(statedic):
    """
    given node state dictionary (from find_nodeState_extend())
    return all positive nodes index
    all positive = Ipre + Isym + Iasym + H
    """
    pos = []
    select = ['I_pre','I_sym','I_asym','H']
    for i in select:
        lst = statedic[i]
        pos.extend(lst)
        
    return pos


def DicCollection(G,individual_ageGroups,households):
    """
    take extended model's network (mode.G)
    this func finds out degree centrality, household size
    nodes' age groups for each node and create a data frame

    also takes individual_ageGroups and households
    they are returned from the network generator
    (generate_demographic_contact_network(....))

    return a dataframe
    index = node's index
    col = nodes' properties
    """
    from collections import defaultdict
    dfd = defaultdict(list)

    degreeC = networkx.degree_centrality(G)
    eigenC = networkx.eigenvector_centrality_numpy(G)

    #node's index number
    idx = range(len(G))

    age = individual_ageGroups
    ageDic = dict(zip(idx,age))


    houseDic = {}
    for x in range(len(households)):
        sz = households[x]['size']
        index = households[x]['indices']
        for a in index:
            houseDic[a] = sz

    for d in (degreeC, eigenC,ageDic,houseDic): 
        for key, value in d.items():
            dfd[key].append(value)

    col = ['DegreeCentrality','EigenCentrality','Ages','HouseholdSize'] 
    df = pd.DataFrame.from_dict(dfd, orient='index',columns = col)
    return df



def clustering(df, eps = 0.15, min_sample = 2):
    """ 
    find clusters in given dataframe 
    (from DicCollection method)

    this func utilizes gower package to calculate similarity
    then uses sklearn.cluster's DBSCAN to calculate clusters

    **you need to install packages for this func

    input: df: df from DicCollection method
         (optional) eps: maximum distance between two samples 
         for one to be considered as in the neighborhood of the other.
         (need to find the best eps )

         (optional)min_sample: minimal number of clusters

    return: cluster labels for each node (no change to node's index) (array)
    """

    #calculate gower distance for each 
    import gower
    distanceM = gower.gower_matrix(df)
    
    clustering = DBSCAN(eps=eps, min_samples=min_sample,metric = 'precomputed').fit(distanceM)

    return clustering.labels_


def degreeCluster(DCdic, eps = 1.0005, min_sample=2):
    """
    do clusters solely based on degree centrality

    input: degree centrality dictionary 
                *:DC = networkx.degree_centrality(model.G)
        (optional) eps: maximum distance between two samples 
         for one to be considered as in the neighborhood of the other.
         (need to find the best eps )

         (optional)min_sample: minimal number of clusters

    output: labels for each node (array)
    """

    #convert dic to 2d array
    v = DCdic.items()
    DClst = [list(a) for a in v]

    dcc = DBSCAN(eps=eps, min_samples=min_sample,).fit(DClst)
    return dcc.labels_


def extendedNet(data,**kwargs):
    """
    this is a pipeline function 
    it creates model and simulate for given days
    
    the disease model parameter is based on the simple covid
    can refer to this website:
    #parameters doc see: https://github.com/ryansmcgee/seirsplus/wiki/ExtSEIRSNetworkModel-class#
    
    
    input: 
        N: number of ppl in network
        
    output:
        baseline model, individualAgeGroup,Households,disease model, all positive node index
    """
    
    df = pd.read_csv(data)
    os.makedirs("outdir", exist_ok = True)

    for index,row in df.iterrows():
        N = row['N']
        days = row['days']



    #generate network
    demographic_graphs, individual_ageGroups, households = generate_demographic_contact_network(
                                                            N=N, demographic_data=household_country_data('US'))
    baselineG   = demographic_graphs['baseline']
    
    latent_p = 3 #virus latent period
    presym_p = 2.5
    infec_p = 7 #infection period
    #create a model with asym people(a) and their infection(beta) is less than symp people 

    model = ExtSEIRSNetworkModel(G=baselineG, p=0.2, beta=0.45, sigma=1/latent_p, lamda=1/presym_p, gamma=1/infec_p,
                             a=0.25, beta_asym=0.3, gamma_asym=1/6.5, h=0.1, eta=1/11.0, gamma_H=1/22.0, f=0.1, mu_H=1/22.0,
                             initE=100)
    
    #run models with the given time
    model.run(T=days)
    
    #find all positive node index
    state = find_nodeState_extend(model.G,model)
    pos = posNode(state)
    
    Dics = DicCollection(baselineG,individual_ageGroups,households)
    Dics.to_csv('report/extended_networkFeatureResult.csv', index = False)

    return baselineG,individual_ageGroups, households,model,pos


def transformAge(netDF):
    """
    transform age col to numerical col
    
    input:network dataframe
    output: cleaned dataframe
    """
    
    #clean categorical data
    ageIx = netDF['Ages'].unique()
    agezip = dict(zip(ageIx,range(len(ageIx))))
    
    netDF = netDF.replace({"Ages":agezip})
    
    return netDF

    

    



