import sys
import pickle
import networkx as nx
import math
import numpy as np
import gzip
import random
import warnings
import pandas as pd

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
    
    print("Reading data ...")
    # read ground truth data and graph g
    g = pickle.load(gzip.open('phishing_url/data/graph.gzpickle', 'rb')) 
    url_truth = pd.read_csv("phishing_url/data/subset_of_data.csv", names=['url', 'label'], skiprows=1)
    data = url_truth['url'].to_list()
    has_ground_truth  = set(data)
    print("Done ...")


    N_FOLDS = 2
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
                relevant_training.add(node)
            elif  url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "good":
                irrelevant_training.add(node)
            else:
                print("error: ground truth error")

        
        # initilize node label
        for node in g.nodes():
            g.nodes[node]['label'] = None
            g.nodes[node]['best_label'] = -1
            g.nodes[node]['data_cost'] = [0.5, 0.5]

            # msg box is a dict
            g.nodes[node]['msgbox'] = {}
            g.nodes[node]['msg_comp'] = [0, 0]
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
        print(training_set)

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
        

       
        
        

if __name__ == '__main__':
    main()
