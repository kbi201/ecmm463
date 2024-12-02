import pickle
import networkx as nx
import numpy as np
import gzip
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from network_construction import add_to_graph, segment_url, save_nodes_and_embeddings


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



class prediction:
    def __init__(self, type_emb, type_sim, type_compat, compat_threshold1, compat_threshold2, N_FOLDS, max_epochs):
            # type_compat = 'table1', 'table2', 'table3'
            # type_emb = None, 'deepwalk', 'node2vec', 'doc2vec', 'word2vec'
            # compat_threshold1, 2 = None, 0.3, 0.5, 0.7
            # type_sim = None, 'rbf', 'minmax', 'cos'
            # ct_type = 'ct1_2', 'ct1_3', 'ct1_4', 'ct2', 'ct3_2', 'ct3_3', 'ct3_4'
            self.type_emb = type_emb
            self.type_sim = type_sim
            self.type_compat = type_compat
            self.compat_threshold1 = compat_threshold1
            self.compat_threshold2 = compat_threshold2
            self.N_FOLDS = N_FOLDS
            self.max_epochs = max_epochs
            self.g = None


    def test_evasion(self, g, evasion_set):

        #print("start")
        for test_url in evasion_set:
            add_to_graph(g, test_url)

        save_nodes_and_embeddings(g)
        print("Test nodes added to Graph")
        

    """
    we are using dataset found in  https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/refs/heads/master/data/data.csv
    this does not use community truth but are labelled good/bad
        - we therefore remove community truths as wont be used
    """
    def main(self):
        
        print("Reading data ...")
        # read ground truth data and graph g
        g = pickle.load(gzip.open('phishing_url/data/graph.gzpickle', 'rb')) 
        self.g = g
        url_truth = pd.read_csv("phishing_url/data/subset_of_data.csv", names=['url', 'label'], skiprows=1)
        data = url_truth['url'].to_list()
        has_ground_truth  = set(data)
        print("Done ...")


        kf = KFold(n_splits=self.N_FOLDS, shuffle=True)
        precision_sum = float(0)
        recall_sum = float(0)
        f1score_sum = float(0)
        accuracy_sum = float(0)

        print("Starting {}-fold cross-validation".format(self.N_FOLDS))
        for train, test in kf.split(data): # iterating through each fold

            # spliting into traniing and testing set for each fold
            training_set = set(np.array(data)[train])
            test_set = set(np.array(data)[test])

            # split training&test set to each class(rel/irrel)
            relevant_training = set()
            irrelevant_training = set()
            relevant_test = set()
            irrelevant_test = set()


            # node == url in the training set
            """
            
            """
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

                # labelling training set and givinginit potentials
                if node in training_set:
                    
                    domain_node = node
                    # graph formatted to include https ../
                    if not domain_node.startswith(("http://", "https://")): 
                        domain_node = "http://" + node

                    g.nodes[domain_node]['label'] = 1 if url_truth.loc[url_truth['url'] == node, 'label'].values[0] == "bad" else 0
                    if g.nodes[domain_node]['label'] == 1:      # malicious
                        g.nodes[domain_node]['data_cost'] = [0.99, 0.01] # we know is malicious so start at 0.99, 0.1
                        mal+=1
                    elif g.nodes[domain_node]['label'] == 0:    # benign
                        g.nodes[domain_node]['data_cost'] = [0.01, 0.99]
                        bn+=1

            print(mal,bn) # show split of malicious and benign nodes


            """ 
            SETTING distances for all edges
            so it looks like embeddings are loaded in seperately from the graph rather than embedding graph nodes themselves?, looks optional however
            """
            if self.type_emb == 'None':
                self.type_emb = None

            if self.type_emb != None:

                with gzip.open("phishing_url/data/graph_embeddings.emb.gzpickle", 'rb') as f:
                    emb = pickle.load(f)

                min_dist = float("inf")
                max_dist = -float("inf")
                
                """
                calculating distance of nodes which will later be used to calc similarity in edge potentails, weher that is equclidian distance etc ..
                """
                for edge in g.edges():

                    if self.type_sim == 'minmax':
                        # euclidean distance
                        g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                        if g.edges[edge]['distance'] > max_dist:
                            max_dist = g.edges[edge]['distance']
                        if g.edges[edge]['distance'] < min_dist:
                            min_dist = g.edges[edge]['distance']
                    elif self.type_sim == 'cos':
                        # cosine similarity
                        g.edges[edge]['sim'] = (np.dot(emb[edge[0]], emb[edge[1]]) / (np.linalg.norm(emb[edge[0]]) * np.linalg.norm(emb[edge[1]])))
                        g.edges[edge]['distance'] = 1 - g.edges[edge]['sim']
                    elif self.type_sim == 'rbf':
                        # euclidean distance
                        g.edges[edge]['distance'] = np.linalg.norm(emb[edge[0]] - emb[edge[1]])
                        # rbf sim (see: https://en.wikipedia.org/wiki/Radial_basis_function_kernel )
                        g.edges[edge]['sim'] = np.exp((-1.0 / 2.0) * np.power(g.edges[edge]['distance'], 2.0))

                if self.type_sim == 'minmax':
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
            for epoch in range(self.max_epochs):
                precision = float(0)
                recall = float(0)
                f1score = float(0)
                accuracy = float(0)

                #visualise_graph(g)
                self.step(g)
                print("Iteration: {} MAP: {}".format(epoch + 1, self.MAP(g)))
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

        avg_precision = precision_sum / self.N_FOLDS
        avg_recall = recall_sum / self.N_FOLDS
        avg_f1 = f1score_sum / self.N_FOLDS
        avg_acc = accuracy_sum / self.N_FOLDS

        print("Averaged precision: {:.6}".format(avg_precision))
        print("Averaged recall: {:.6}".format(avg_recall))
        print("Averaged F1 score: {:.6}".format(avg_f1))
        print("Averaged accuracy: {:.6}".format(avg_acc))

        print("End: " + str(datetime.now()))



        ## conduct test on unseen evasions
        #self.test_evasion(g)

        return avg_precision, avg_recall, avg_f1, avg_acc

    """
    This function propogates message
    --------------------------------

    each step send a message from a node to its neighbours
        - dont sent a message to a labelled node as obesrved variables do not recieve messages
        - if sending from a labelled node use _send_msg_label, else _send_msg where the largest diff comes from how the message is calculated
    """
    def step(self, G):
        for n in tqdm(G.nodes(), desc="Propagate from vertices with label", mininterval=0.5): # tqdm inits a progress bar
            if G.nodes[n]['label'] != None:
                for nbr in G.neighbors(n):
                    # do not propagate to nodes with label
                    if G.nodes[nbr]['label'] == None:

                        #print("HIDDEN", nbr)
                        self._send_msg_label(G, n, nbr)
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
                        self._send_msg(G, n, nbr)
        #for n in tqdm(G.nodes(), desc="Compiling message boxes 2", mininterval=0.5):
        #    G.nodes[n]['msg_comp'] = [0, 0]
        #    for nbr in G.neighbors(n):
        #        G.nodes[n]['msg_comp'][0] += G.nodes[n]['msgbox'][nbr][0]
        #        G.nodes[n]['msg_comp'][1] += G.nodes[n]['msgbox'][nbr][1]

    """
    calculates the message for hidden variables/nodes
    """
    def _min_sum(self, G, _from, _to):
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
            if self.type_compat == 'table1':
                # original (we think this version is for sum-product...)
                #p_not_related += 0.5 + eps if i == 0 else 0.5 - eps
                #p_related += 0.5 - eps if i == 0 else 0.5 + eps

                """ using Polonium based Heurisitic. chose of heuristic is likely dependent on similarity measures used"""
                p_not_related += 0.5 - eps if i == 0 else 0.5 + eps 
                p_related += 0.5 + eps if i == 0 else 0.5 - eps
            elif self.type_compat == 'table2':
                # original (this version works only when table2 && cos)
                #p_not_related += 0 if i == 0 else 1 - G[_from][_to]['distance']
                #p_related += 1 - G[_from][_to]['distance'] if i == 0 else 0
                #p_not_related += 0 if i == 0 else G[_from][_to]['sim']
                #p_related += G[_from][_to]['sim'] if i == 0 else 0
                p_not_related += 0 if i == 0 else G[_from][_to]['distance']
                p_related += G[_from][_to]['distance'] if i == 0 else 0
            elif self.type_compat == 'table3':
                # original (our sim are similarities -> same = 1 / completely different = 0)

                """
                EDGE POTENTIALS USING COMPATIBILY MATRIX DEFINED IN THE PAPER
                -------------------------------------------------------------
                        Phishy                       Benign
                Phishy  min(ths+, 1 - sim(x, y))     max(ths−, sim(x, y))
                Benign  max(ths−, sim(x, y))         min(ths+, 1 - sim(x, y))
                """
                p_not_related += np.min([self.compat_threshold1, 1 - G[_to][_from]['sim']]) if i == 0 else np.max([self.compat_threshold2, G[_to][_from]['sim']])
                p_related += np.max([self.compat_threshold2, G[_to][_from]['sim']]) if i == 0 else np.min([self.compat_threshold1, 1 - G[_to][_from]['sim']])
                
            new_msg[i] = min(p_not_related, p_related)
            #print(new_msg)

        # Normalization
        # new_msg = np.exp(new_msg) / np.sum(np.exp(new_msg))

        return new_msg

    """
    This function propogates messages from labelled nodes
    if from node is maliciious then msg = [1, 0] else benign is [0, 1]
    """
    def _send_msg_label(self, G, _from, _to):
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


    def _send_msg(self, G, _from, _to, ):
        # label not given
        msg = self._min_sum(G, _from, _to)

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
    def MAP(self, G):
        n_wrong_label = 0
        n_correct_label = 0

        # remember the objective is to minimise costs
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

                # if improper labelling do not consider
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
    def visualise_graph(g):
        pos = nx.spring_layout(g, seed=42)  # Define layout for better aesthetics

        # Plot the nodes with different colors based on their type (URL, Domain, Word)
        url_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'URL']
        domain_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'Domain']
        word_nodes = [n for n, attr in g.nodes(data=True) if attr.get('type') == 'Word']

        # Draw nodes
        nx.draw_networkx_nodes(g, pos, nodelist=url_nodes, node_size=600, node_color="skyblue", label="URLs")
        nx.draw_networkx_nodes(g, pos, nodelist=domain_nodes, node_size=300, node_color="lightgreen", label="Domains")
        nx.draw_networkx_nodes(g, pos, nodelist=word_nodes, node_size=150, node_color="salmon", label="Words")

        # Draw edges with weights
        edge_weights = [g[u][v].get('distance', 1) for u, v in g.edges()]
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), width=edge_weights, alpha=0.7, edge_color="gray")

        """

        # Draw labels (for nodes)
        node_labels = {}
        for node, attr in g.nodes(data=True):
            # Only display node labels if the node has a 'label' attribute
            if 'label' in attr:
                node_labels[node] = attr['label']
        
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=10, font_color="black")
        """

        # Optional: Display edge weights as labels on the graph
        #edge_labels = {(u, v): round(g[u][v].get('distance', 0), 2) for u, v in g.edges()}
        #nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Graph Visualization with Node and Edge Labels")
        plt.legend(scatterpoints=1, loc="upper right", fontsize=10)
        plt.axis('off')  # Turn off the axis
        plt.show()



if __name__ == '__main__':

    # parameter space
    grid = {
        'type_sim': ['rbf', 'minmax', 'cos'],
        'type_compat': ['table1', 'table2', 'table3'],
        'compat_threshold1': [0.3, 0.5, 0.7],
        'compat_threshold2': [0.3, 0.5, 0.7],
        'N_FOLDS': [3 ,5, 7],
        'max_epochs': [3, 5, 7]
    }
    # not including emb in param space as graph constructed outside of scope
    type_emb = 'node2vec' 

    # Random grid search
    random_trails = 4  # Number of random configurations to try
    best_score = float("-inf")
    best_params = None
    best_model = None
    config_params = None

    for _ in range(random_trails):

        params = {key: np.random.choice(values) for key, values in grid.items()} # select random param for key
        print(params)
        
        # train and eval using random param config
        p = prediction(type_emb, type_sim= params["type_sim"], type_compat= params["type_compat"], compat_threshold1= params["compat_threshold1"],
                        compat_threshold2= params["compat_threshold2"], N_FOLDS= params["N_FOLDS"], max_epochs= params["max_epochs"])
        
        avg_precision, avg_recall, avg_f1, avg_acc = p.main()
        score = (avg_precision +  avg_recall + avg_f1 +  avg_acc) / 4 # using average of all metrics to calc score
        
        if score > best_score:
            best_score = score
            best_params = (avg_precision, avg_recall, avg_f1, avg_acc)
            config_params = params
            best_model = p

    print(f"Best Score: {best_score}")
    print(f"Best config: {config_params}")
    print(f"Best Average performance: {best_params}")


    ## Now using the best performing graph, test evasions 
    # containing evasion instances m1, m2, ... m7
    test_set = [
        {"url": "015fb31.netsolhost.com/css/ballet-guitars-notfree-mp3", "truth": "1"}, #M1 URL -> benign
        {"url": "gulsproductionscar.com/bosstweed/notphish/good.html", "truth": "1"}, #M2 Path stgin -> benign
        {"url": "gulsproductions.com/css/balmoral-hat-green?personID=I4920&tree=ncshawfamily", "truth": "1"}, #M3 Query string -> beniegn
        {"url": "032255hellooo.com/lincolnhomepage?flow_id=2000&870470=33440/case_id=17188", "truth": "1"}, #M4 domain, path -> benign
        {"url": "032255hellooo.com/css/ballet-guitars-notfree-mp3?personID=I4920&tree=ncshawfamily", "truth": "1"}, #M5 domain and query -> benign
        {"url": "015fb31.netsolhost.com/merchant2/merchant.mvc?Screen=CTGY&Store_Code=BC&Category_Code=CP", "truth": "1"}, # M6 path n Query String -> beneign
        {"url": "01453car.com/", "truth": "0"},
        {"url": "015fb31.netsolhost.com/bosstweed/notphish/good.html", "truth": "0"},
        {"url": "02bee66.netsolhosttrustme.com/lincolnhomepage/", "truth": "0"},
        {"url": "02ec0a3.netsolhosttest.com/getperson.php?personID=I4920&tree=ncshawfamily", "truth": "0"},
        {"url": "032255hellooo.com/", "truth": "0"}
    ]

    best_model.test_evasion(best_model.g, test_set)
    best_model.main() # rerun belief propogation
    count = 0

    for row in test_set:
        if not row["url"].startswith(("http://", "https://")): row["url"] = "http://" + row["url"]
        print(row["url"])

        if row["url"] in best_model.g:
            model_prediction = best_model.g.nodes[row["url"]]["best_label"]
            print(f"Prediction  {model_prediction} \n")
            if model_prediction == int(row["truth"]): count += 1

    print( count / len(test_set) * 100 )


    
    # check labels


