import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 
import math

R0 = 2.38
SEVERE_SYMPTOM_RATE = 0.138
CRITICAL_MORTALITY = 0.405 

# node attributes for later network generating
Default_Node_Attr = {'S': 1, "E": 0, "I": 0, "R": 0, "D": 0}

def node_generate(N):
    """ 
    generate network
    ======================
    
    N: number of nodes 
    
    """
    
    g=nx.Graph()
    g.add_nodes_from(range(N))
    
    # get default attributes
    idx = list(range(N))
    dic = dict()
    for x in idx:
        dic[x] = Default_Node_Attr
        
    # set all nodes to default attributes 
    nx.set_node_attributes(g, dic)
    return g

def randomdize_relationship(start, end, G):
    """
    create random relationships
    given a range of number of friends each person have
    
    generate edges from this range randomly
    ==========================================
    
    start = min number of friends/edges a person has (inclusive)
    end = max number of friends/edges a person has (exclusive)
    
    end should not greater than the total nodes -1(self)
    
    G = the network
    """
    
    if end > len(G.nodes)-1: 
        print('Number of realtionships per person exceeds number of person in the network!')
        return 
    
    friend_range = list(range(start,end))
    for a in G.nodes:
        # potential edge node lists without itself
        potential_friends = [x for x in G.nodes if x != a]
        
        # randomly pick number of friends this person has
        n = random.choice(friend_range)
        
        # pick random nodes
        fri = random.sample(potential_friends, n)
        
        
        # add edges with random generated initilized weights (range = 100)
        # higher weights means how close this friend is with this person
        # higher weights means more close contacts with this friend
        for i in range(len(fri)):
            weight = random.choice(range(100))
            G.add_edge(a,potential_friends[i], weight = weight)
    return G


def initialize_I(n, G):
    """
    randomly select n people infected in the network
    
    n = number of initial infected people
    
    return NONE
    """
    nodes = G.nodes
    select = random.sample(nodes, n)
    
    #reset their attributes
    attrs = {'S': 0, "E": 0, "I": 1, "R": 0, "D": 0}
   
    dic = dict()
    for x in select:
        dic[x] = attrs
        
    # set all nodes to default attributes 
    nx.set_node_attributes(G, dic)


def get_infected(G):
    """
    BASED on R0, 1 person avg spread to 2 people
    those people are selected based on how close the relationship is with infected person
    """
   
    
    #find how many nodes are infected
    R0 = 2
    allnode = len(G.nodes())
    infected = [x for x, y in G.nodes(data = True) if y['I'] == 1]
     
    
    for x in infected:
        #find all ppl has relationship with this infected person
        related = [d for d in nx.all_neighbors(G,x)]
        #find all edge weights (how close they are)
        weights = [G[x][a]['weight'] for a in related]
        
        ziped = list(zip(related,weights))
        
        #sort by values
        sot = sorted(ziped, key=lambda x: float(x[1]), reverse=True)
        
        count = 0
        
        for a in sot:
            key = a[0]
            if G.nodes[key]['I'] == 0:
                if (G.nodes[key]['S'] == 1) or (G.nodes[key]['E'] == 1):
                
                    #reset the attribute
                    G.nodes[key]['S'] = 0
                    G.nodes[key]['I'] = 1
                    count += 1
                    
                    if count >= 2: break
            
    return G



def get_resolved(G):
    """
    determine the results of infected person
    
    within severe symptom rate: 
        within mortality rate: I to D 
    others: reamin in I to R
    ================================
    G = network
    
    return = network with updated attribute I, R and D
    """
    for x in G.nodes():
        #check is this node in infected state
        if G.nodes[x]['I'] != 1: continue

        ram = random.choice(range(100))

        if ram < (SEVERE_SYMPTOM_RATE * 100):
            ram2 = random.choice(range(100))
            if ram2 < (CRITICAL_MORTALITY):
                # dead end
                #reset the attribute
                G.nodes[x]['I'] = 0
                G.nodes[x]['D'] = 1
        else:
            #recovered
            G.nodes[x]['I'] = 0
            G.nodes[x]['R'] = 1
            
    return G




def get_susceptible(G):
    """
    currently it is a very simple model
    just putting all R back to S
    
    later can incorporate immunity duration parameter
    ===================================
    
    TBC=====>
    """
    for x in G.nodes():
        #check is this node in infected state
        if G.nodes[x]['R'] == 1:
            G.nodes[x]['S'] = 1
            G.nodes[x]['R'] = 0
    return G


def disease_spread(t,n,IF):
    """
    disease spread among population in t
    
    t = number of units of the time period
    eg: t = 3, disease spreaded in 3 time units
    undergoing all funcs above
    
    n = number of ppl in network
    
    IF = initial infected number of ppl
    """
    
    #generate network with n ppl
    G = node_generate(n)
    
    # 3 to 20 friends per person
    upper = 20
    if n <= 20:
        #if 20 friends greater than n we have
        #pick 25% as our upper bound
        upper = int(n * 0.25)
    G = randomdize_relationship(1,upper,G)
    
    #initialize infection
    initialize_I(IF, G)
    
    for i in range(t):
        #G = get_exposed(G)
        G = get_infected(G)
        
        if (i+1) % 2 == 0:
            G = get_resolved(G)
            G = get_susceptible(G)
        
    return G


def all_pos(G):
    size = [x for x, y in G.nodes(data = True) if y['I'] == 1]
    return size













