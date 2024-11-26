import sys
import pickle
import networkx as nx
import math
import numpy as np
import gzip
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import KFold
from datetime import datetime
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# from sklearn.metrics.pairwise import rbf_kernel
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

"""
we are using dataset found in  https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/refs/heads/master/data/data.csv
this does not use community truth but are labelled good/bad
    - we therefore remove community truths as wont be used
"""
def main():

    # type_compat = 'table1', 'table2', 'table3'
    # type_emb = None, 'deepwalk', 'node2vec', 'doc2vec', 'word2vec'
    # compat_threshold1, 2 = None, 0.3, 0.5, 0.7
    # type_sim = None, 'rbf', 'minmax', 'cos'
    # ct_type = 'ct1_2', 'ct1_3', 'ct1_4', 'ct2', 'ct3_2', 'ct3_3', 'ct3_4'
    type_emb = 'node2vec'
    type_sim = 'cos'
    type_compat = 'table2'
    compat_threshold1 = 0.7
    compat_threshold2 = 0.7
    
    print("Reading data ...")
    # read ground truth data and graph g
    g = pickle.load(gzip.open('phishing_url/data/graph.gzpickle', 'rb')) 
    url_truth = pd.read_csv("phishing_url/data/subset_of_data.csv", names=['url', 'label'], skiprows=1)
    data = url_truth['url'].to_list()
    has_ground_truth  = set(data)
    print("Done ...")


    N_FOLDS = 3
    max_epoch = 3

    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    precision_sum = float(0)
    recall_sum = float(0)
    f1score_sum = float(0)
    accuracy_sum = float(0)

    print("Starting {}-fold cross-validation".format(N_FOLDS))
    for train, test in kf.split(data): # iterating through each fold

        # spliting into traniing and testing set for each fold
        training_set = set(np.array(data)[train])
        test_set = set(np.array(data)[test])

        # split training&test set to each class(rel/irrel)
        relevant_training = set()
        irrelevant_training = set()
        relevant_test = set()
        irrelevant_test = set()

        #print(training_set)

        # node == url in the training set
        for node in training_set:

            # checking df for circumstances where urls match, returning label field, with first occurence (we assume one occurence of each url anyways)
            if url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "bad":
                relevant_training.add(node)
            elif  url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "good":
                irrelevant_training.add(node)
            else:
                print("error: ground truth error")

        for node in test_set:
            if url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "bad":
                relevant_test.add(node)
            elif  url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "good":
                irrelevant_test.add(node)
            else:
                print("error: ground truth error")

        
        # initilize node label
        for node in g.nodes():
            g.nodes[node]['label'] = None
            g.nodes[node]['best_label'] = -1
            g.nodes[node]['data_cost'] = [0.5, 0.5] # default beliefs about being phishy or beneign following polonium heurisitic

            # msg box is a dict
            g.nodes[node]['msgbox'] = {} # this will maintain a list of each message being passed to each neighbour
            g.nodes[node]['msg_comp'] = [0, 0] # initialising iysmessages to be passed onto the neighbours
            for nbr in list(g.neighbors(node)):
                g.nodes[node]['msgbox'][nbr] = [0, 0] 
   
        """
        for each node belonging in the training set we want to give it a label where 
        according to the paper, hidden variables are not known entities thus do not have
        labels

        note that instead of 0,1, our dataset used good and bad labels. so bad = 1, good = 0
        """
        
        mal = 0 # malicious counter
        bn = 0 # beniegn counter
        #print(training_set)

        for node in has_ground_truth:
            if node in training_set:
                
                domain_node = node
                # graph formatted to include https ../
                if not domain_node.startswith(("http://", "https://")): 
                    domain_node = "http://" + node

                g.nodes[domain_node]['label'] = 1 if url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "bad" else 0
                if g.nodes[domain_node]['label'] == 1:      # malicious
                    g.nodes[domain_node]['data_cost'] = [0.99, 0.01]
                    mal+=1
                elif g.nodes[domain_node]['label'] == 0:    # benign
                    g.nodes[domain_node]['data_cost'] = [0.01, 0.99]
                    bn+=1

        print(mal,bn) # show split of malicious and benign nodes


        """ 
        SETTING distances for all edges
        so it looks like embeddings are loaded in seperately from the graph rather than embedding graph nodes themselves?, looks optional however
        """
        if type_emb == 'None':
            type_emb = None

        if type_emb != None:

            with gzip.open("phishing_url/data/graph_embeddings.emb.gzpickle", 'rb') as f:
                emb = pickle.load(f)

            min_dist = float("inf")
            max_dist = -float("inf")
            
            # calculating edge potentials using similarity approach providied to calculate distance
            for edge in g.edges():

                if type_sim == 'minmax':
                    # euclidean distance
                    g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                    if g.edges[edge]['distance'] > max_dist:
                        max_dist = g.edges[edge]['distance']
                    if g.edges[edge]['distance'] < min_dist:
                        min_dist = g.edges[edge]['distance']
                elif type_sim == 'cos':
                    # cosine similarity
                    g.edges[edge]['sim'] = (np.dot(emb[edge[0]], emb[edge[1]]) / (np.linalg.norm(emb[edge[0]]) * np.linalg.norm(emb[edge[1]])))
                    g.edges[edge]['distance'] = 1 - g.edges[edge]['sim']
                elif type_sim == 'rbf':
                    # euclidean distance
                    g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                    # rbf sim (see: https://en.wikipedia.org/wiki/Radial_basis_function_kernel )
                    g.edges[edge]['sim'] = np.exp((-1.0 / 2.0) * np.power(g.edges[edge]['distance'], 2.0))

            if type_sim == 'minmax':
                for edge in g.edges():
                    g.edges[edge]['sim'] = 1 - np.divide((g.edges[edge]['distance'] - min_dist), max_dist-min_dist)
            
            print("embedding done")
        # if type_emb not provided...
        else:
            # set initial messages
            for edge in g.edges():
                #g.edges[edge]['msg'] = [0, 0]
                g.edges[edge]['distance'] = 1.0
                g.edges[edge]['sim'] = 0.5

        print("Done.")

        """
        This is the actual belief propogation section of the algorithm, which loops for max_epochs
        number of iterations performing the following:
            - 
        """
        for epoch in range(max_epoch):
            precision = float(0)
            recall = float(0)
            f1score = float(0)
            accuracy = float(0)

            #visualisie_graph(g)
            step(g, type_compat, compat_threshold1=compat_threshold1, compat_threshold2=compat_threshold2)
            print("Iteration: {} MAP: {}".format(epoch + 1, MAP(g)))
            #visualisie_graph(g)


            #print(relevant_test, "\n")
            #print(g.nodes)

            relevant_correctness = 0
            relevant_incorrectness = 0
            for i in relevant_test:
                if not i.startswith(("http://", "https://")): 
                    i = "http://" + i
                    
                if g.nodes[i]['best_label'] == 1:
                    relevant_correctness += 1
                else:
                    relevant_incorrectness += 1

            irrelevant_correctness = 0
            irrelevant_incorrectness = 0
            for i in irrelevant_test:
                if not i.startswith(("http://", "https://")): 
                    i = "http://" + i
                    

                if g.nodes[i]['best_label'] == 0:
                    irrelevant_correctness += 1
                else:
                    irrelevant_incorrectness += 1

            print("rel_cor: " + str(relevant_correctness))
            print("rel_incor: " + str(relevant_incorrectness))
            print("irrel_cor: " + str(irrelevant_correctness))
            print("irrel_incor: " + str(irrelevant_incorrectness))

            print("Relevant Accuracy: {:.6}".format(relevant_correctness / (relevant_correctness + relevant_incorrectness)))
            print("Irrelevant Accuracy: {:.6}".format(irrelevant_correctness / (irrelevant_correctness + irrelevant_incorrectness)))

            if (relevant_correctness + irrelevant_incorrectness) == 0:
                precision = float(0)
            else:
                precision = relevant_correctness / (relevant_correctness + irrelevant_incorrectness)
            print("Precision: {:.6}".format(precision))

            if (relevant_correctness + relevant_incorrectness) == 0:
                recall = float(0)
            else:
                recall = relevant_correctness / (relevant_correctness + relevant_incorrectness)
            print("Recall: {:.6}".format(recall))

            if (precision + recall) == 0:
                f1score = float(0)
            else:
                f1score = 2 * precision * recall / (precision + recall)
            print("F1 score: {:.6}".format(f1score))

            accuracy = (relevant_correctness + irrelevant_correctness) / (relevant_correctness + relevant_incorrectness + irrelevant_correctness + irrelevant_incorrectness)
            print("Accuracy: {:.6}".format(accuracy))

        precision_sum += precision
        recall_sum += recall
        f1score_sum += f1score
        accuracy_sum += accuracy
        print()
    
    print("Done.")
    print()

    print("Averaged precision: {:.6}".format(precision_sum / N_FOLDS))
    print("Averaged recall: {:.6}".format(recall_sum / N_FOLDS))
    print("Averaged F1 score: {:.6}".format(f1score_sum / N_FOLDS))
    print("Averaged accuracy: {:.6}".format(accuracy_sum / N_FOLDS))

    print("End: " + str(datetime.now()))



"""
This function propogates message
--------------------------------

each step send a message from a node to its neighbours
    - dont sent a message to a labelled node as obesrved variables do not recieve messages
    - if sending from a labelled node use _send_msg_label, else _send_msg where the largest diff comes from how the message is calculated
"""
def step(G, type_compat, compat_threshold1=None, compat_threshold2=None):
    for n in tqdm(G.nodes(), desc="Propagate from vertices with label", mininterval=0.5): # tqdm inits a progress bar
        if G.nodes[n]['label'] != None:
            for nbr in G.neighbors(n):
                # do not propagate to nodes with label
                if G.nodes[nbr]['label'] == None:

                    #print("HIDDEN", nbr)
                    _send_msg_label(G, n, nbr)
    #for n in tqdm(G.nodes(), desc="Compiling message boxes 1", mininterval=0.5):
    #    G.nodes[n]['msg_comp'] = [0, 0]
    #    for nbr in G.neighbors(n):
    #        G.nodes[n]['msg_comp'][0] += G.nodes[n]['msgbox'][nbr][0]
    #        G.nodes[n]['msg_comp'][1] += G.nodes[n]['msgbox'][nbr][1]
    for n in tqdm(G.nodes(), desc="Propagate from vertices without label", mininterval=0.5):
        if G.nodes[n]['label'] == None:
            for nbr in G.neighbors(n):
                # do not propagate to nodes with label
                if G.nodes[nbr]['label'] == None:
                    _send_msg(G, type_compat, n, nbr, compat_threshold1=compat_threshold1, compat_threshold2=compat_threshold2)
    #for n in tqdm(G.nodes(), desc="Compiling message boxes 2", mininterval=0.5):
    #    G.nodes[n]['msg_comp'] = [0, 0]
    #    for nbr in G.neighbors(n):
    #        G.nodes[n]['msg_comp'][0] += G.nodes[n]['msgbox'][nbr][0]
    #        G.nodes[n]['msg_comp'][1] += G.nodes[n]['msgbox'][nbr][1]

"""
calculates the message for hidden variables/nodes
"""
def _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold2):
    eps = 0.001

    new_msg = [0] * 2
    for i in range(2):  # we only have 2 labels so far
        fromnode = G.nodes[_from]

        # initialize
        # related => label 1
        # not related => label 0
        p_not_related = 0
        p_related = 0

        # data cost
        #p_not_related += math.log(1 - fromnode['data_cost'][0])
        #p_related += math.log(1 - fromnode['data_cost'][1])
        p_not_related += fromnode['data_cost'][0]
        p_related += fromnode['data_cost'][1]

        #for nbr in G.neighbors(_from):
        #    if nbr == _to:
        #        continue
        #    p_not_related += fromnode['msgbox'][nbr][0]
        #    p_related += fromnode['msgbox'][nbr][1]
        p_not_related += fromnode['msg_comp'][0] - fromnode['msgbox'][_to][0]
        p_related += fromnode['msg_comp'][1] - fromnode['msgbox'][_to][1]

        # smoothness cost
        if type_compat == 'table1':
            # original (we think this version is for sum-product...)
            #p_not_related += 0.5 + eps if i == 0 else 0.5 - eps
            #p_related += 0.5 - eps if i == 0 else 0.5 + eps
            p_not_related += 0.5 - eps if i == 0 else 0.5 + eps
            p_related += 0.5 + eps if i == 0 else 0.5 - eps
        elif type_compat == 'table2':
            # original (this version works only when table2 && cos)
            #p_not_related += 0 if i == 0 else 1 - G[_from][_to]['distance']
            #p_related += 1 - G[_from][_to]['distance'] if i == 0 else 0
            #p_not_related += 0 if i == 0 else G[_from][_to]['sim']
            #p_related += G[_from][_to]['sim'] if i == 0 else 0
            p_not_related += 0 if i == 0 else G[_from][_to]['distance']
            p_related += G[_from][_to]['distance'] if i == 0 else 0
        elif type_compat == 'table3':
            # original (our sim are similarities -> same = 1 / completely different = 0)
            p_not_related += np.min([compat_threshold1, 1 - G[_to][_from]['sim']]) if i == 0 else np.max([compat_threshold2, G[_to][_from]['sim']])
            p_related += np.max([compat_threshold2, G[_to][_from]['sim']]) if i == 0 else np.min([compat_threshold1, 1 - G[_to][_from]['sim']])
            
        new_msg[i] = min(p_not_related, p_related)
        #print(new_msg)

    # Normalization
    # new_msg = np.exp(new_msg) / np.sum(np.exp(new_msg))

    return new_msg

"""
This function propogates messages from labelled nodes
if from node is maliciious then msg = [1, 0] else benign is [0, 1]
"""
def _send_msg_label(G, _from, _to):
    #print(f"FROM   {G.nodes[_from]}" )
    # if lable is given
    if G.nodes[_from]['label'] == 1:
        msg = [1, 0]
    elif G.nodes[_from]['label'] == 0:
        msg = [0, 1]
    else:
        # ct2 case
        msg = G.nodes[_from]['data_cost']

    to_node = G.nodes[_to]
    #print(msg, to_node)
    #print("\n", to_node['msg_comp'][0], to_node['msgbox'][_from][0])


    # subtract original msg
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
    #print("\n", to_node['msg_comp'][0], to_node['msgbox'][_from][0])

    # add new msg
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]

    # orignal msg := new msg
    to_node['msgbox'][_from] = msg
    #print(f"FROM   {G.nodes[_from]}    TO     {G.nodes[_to]['msg_comp'][0]} \n", )


def _send_msg(G, type_compat, _from, _to, compat_threshold1 = None, compat_threshold2 = None):
    # label not given
    msg = _min_sum(G, _from, _to, type_compat, compat_threshold1, compat_threshold1)

    to_node = G.nodes[_to]
    # subtract original msg from from node
    to_node['msg_comp'][0] -= to_node['msgbox'][_from][0]
    to_node['msg_comp'][1] -= to_node['msgbox'][_from][1]
     
    # add new msg
    to_node['msg_comp'][0] += msg[0]
    to_node['msg_comp'][1] += msg[1]
    # orignal msg := new msg
    to_node['msgbox'][_from] = msg



"""
This function is used to evalatue perfomance of belief propgoation
"""
def MAP(G):
    n_wrong_label = 0
    n_correct_label = 0

    for n in G.nodes():
        #print(G.nodes[n])
        nodedata = G.nodes[n]

        cost_not_related = 0
        cost_related = 0

        # data costs
        cost_not_related += nodedata['data_cost'][0]
        cost_related += nodedata['data_cost'][1]

        # msg comp
        cost_not_related += nodedata['msg_comp'][0]
        cost_related += nodedata['msg_comp'][1]

        if cost_related < cost_not_related:
            nodedata['best_label'] = 1
        else:
            nodedata['best_label'] = 0

        
        #print(cost_related, cost_not_related, nodedata['best_label'], nodedata['label'])

        # as we are only checking labelled nodes, only concerned with url nodes
        if (nodedata['label'] == 1 and nodedata['best_label'] == 0) or (nodedata['label'] == 0 and nodedata['best_label'] == 1) :
            #print("error2: wrong label!")
            n_wrong_label += 1
    
        elif (nodedata['label'] == 1 and nodedata['best_label'] == 1) or (nodedata['label'] == 0 and nodedata['best_label'] == 0):
            n_correct_label += 1
        else:
            pass

    print("# wrong label: " + str(n_wrong_label))
    print("# correct label: " + str(n_correct_label))
    


    """
    energy in this case looks like some measure of distance or metric, whereby the larger indicates 
        - higher datacosts assigned per label
        - more misclassifications as there are more discrepancies between predicted "best label" and curr label
    """
    energy = 0
    for n in G.nodes():
        cur_label = G.nodes[n]['best_label']

        #energy += math.log(1 - G.node[n]['data_cost'][cur_label])
        energy += G.nodes[n]['data_cost'][cur_label]
        for nbr, eattr in G[n].items():
            energy += 0 if G.nodes[nbr]['best_label'] == cur_label else eattr['distance']

    return energy


"""
visualse graph with edge weights and stuff
"""
def visualisie_graph(g):
    pos = nx.spring_layout(g, seed=42)  # Define layout for better aesthetics

    # Plot the nodes with different colors based on their type (URL, Domain, Word)
    url_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'URL']
    domain_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'Domain']
    word_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'Word']

    # Draw nodes
    nx.draw_networkx_nodes(g, pos, nodelist=url_nodes, node_size=500, node_color="skyblue", label="URLs")
    nx.draw_networkx_nodes(g, pos, nodelist=domain_nodes, node_size=500, node_color="lightgreen", label="Domains")
    nx.draw_networkx_nodes(g, pos, nodelist=word_nodes, node_size=500, node_color="salmon", label="Words")

    # Draw edges with weights
    # If the 'distance' attribute is available, use it to adjust edge width
    edge_weights = [g[u][v].get('distance', 1) for u, v in g.edges()]
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), width=edge_weights, alpha=0.7, edge_color="gray")

    # Draw labels (for nodes)
    node_labels = {}
    for node, attr in g.nodes(data=True):
        # Only display node labels if the node has a 'label' attribute
        if 'label' in attr:
            node_labels[node] = attr['label']
    
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=10, font_color="black")

    # Optional: Display edge weights as labels on the graph
    edge_labels = {(u, v): round(g[u][v].get('distance', 0), 2) for u, v in g.edges()}
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Graph Visualization with Node and Edge Labels")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=10)
    plt.axis('off')  # Turn off the axis
    plt.show()



        
        

if __name__ == '__main__':
    main()
