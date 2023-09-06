import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
from tqdm import tqdm
import random
from monte_carlo import MonteCarlo_simulation
from community import community_louvain
import networkx as nx
import community as BGLL
#import community
from collections import Counter
from load import read_graph
from config import *
import pandas as pd

G= pd.read_csv("datasets/soc-dolphins.mtx")

def modularity(G,partition):
#calculate the modularity of a partition in a graph G
    m = G.number_of_edges()
    Q = 0
    for c in set(partition.values()):
        nodes_in_c=[n for n in partition if partition[n]==c ]
        L_c=sum(G.degree[n] for n in nodes_in_c)
        d_c= L_c / (2*m)
        E_c= sum(G.degree[u] for u in nodes_in_c)
        Q += (E_c / (2 * m)) - d_c**2
    return Q
def louvian(G):
   # partition = {n: n for n in G.nodes()}
    partition = community_louvain.best_partition(G)
    return partition
    
    mod= modularity(G,partition)

    while True:
        improved= False
        for node in G.nodes():
            current_community = partition[node]
            best_community= current_community
            best_modularity = mod


            neighbors= list(G.neighbors(node))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                partition[node]=partition[neighbor]
                new_modularity= modularity(G,partition)
                if new_modularity > best_modularity:
                    best_modularity= new_modularity
                    best_community=partition[neighbor]
            if best_community !=current_community:
                partition[node]=best_community
                mod=best_modularity
                improved = True
        if not improved:
            break    
    return partition

#example usage

#ایجاد گراف
#G= nx.Graph()  
#G.add_edges_from([(0,1),(0,2),(1,2),(3,4),(4,5),(3,5),(1,4),(3,1),(3,2)])

#G=pd.read_csv("datasets/soc-dolphins.mtx")
#G=dataset_path
partition =louvian(G)  
print(partition)  

#تقسیم گراف به اجتماع‌ها و تخصیص رنگ‌ها
communities= list(nx.community.greedy_modularity_communities(G))
node_colors= [i for  i , community in enumerate(communities) for _ in range(len(community))]

#رسم گراف
pos= nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_color=node_colors,cmap=plt.get_cmap('viridis'), node_size=200)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_edge_labels(G,pos)
plt.axis('off')
plt.show()