import numpy as np
import json
import networkx as nx
import sys
import random
import copy
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def k_fold_split_edges(gFull, k):
    """
        Split edges into k groups. Return a list of k lists of edges
    """
    pass

def random_split(gFull, sval=0.05, stest=0.05):
    gTrain = nx.Graph()
    gValid = nx.Graph()
    gTest = nx.Graph()

    gTrain.add_nodes_from(gFull.nodes)
    gValid.add_nodes_from(gFull.nodes)
    gTest.add_nodes_from(gFull.nodes)

    all_edges = copy.deepcopy(list(gFull.edges()))
    random.shuffle(all_edges)

    print(len(gFull.nodes))
    print(len(gFull.edges))
        

    N = len(all_edges)
    NValid = int(N * sval)
    NTest = int(N * stest)
    NTrain = N - NValid - NTest

    gTrain.add_edges_from(all_edges[:NTrain])
    gValid.add_edges_from(all_edges[NTrain : NTrain + NValid])
    gTest.add_edges_from(all_edges[NTrain + NValid :])

    print("train edges:", len(gTrain.edges))
    print("valid edges:", len(gValid.edges))
    print("test edges: ", len(gTest.edges))

    print("train nodes:", len(gTrain.nodes))
    print("valid nodes:", len(gValid.nodes))
    print("test nodes:", len(gTest.nodes))

    return gTrain, gValid, gTest

def link_prediction(gTrain, gValid, metric='jc'):

    print(f'Computing {metric}...')

    if metric == 'jc':
        preds_pos = nx.jaccard_coefficient(gTrain, nx.edges(gValid))
        preds_neg = nx.jaccard_coefficient(gTrain, nx.non_edges(gValid))
    elif metric == 'rai':
        preds_pos = nx.resource_allocation_index(gTrain, nx.edges(gValid))
        preds_neg = nx.resource_allocation_index(gTrain, nx.non_edges(gValid))
    elif metric == 'aai':
        preds_pos = nx.adamic_adar_index(gTrain, nx.edges(gValid))
        preds_neg = nx.adamic_adar_index(gTrain, nx.non_edges(gValid))
    elif metric == 'pa':
        preds_pos = nx.preferential_attachment(gTrain, nx.edges(gValid))
        preds_neg = nx.preferential_attachment(gTrain, nx.non_edges(gValid))
    elif metric == 'cnc':
        preds_pos = nx.common_neighbor_centrality(gTrain, nx.edges(gValid), alpha=0.8)
        preds_neg = nx.common_neighbor_centrality(gTrain, nx.non_edges(gValid), alpha=0.8)
    else:
        print("No such metric!")
        return

    y_preds = []
    y_true = []
    for u, v, s in preds_pos:
        y_true.append(1)
        y_preds.append(s)

    size = len(y_true)
    i = 0
    for u, v, s in preds_neg:
        if i >= size:
            break
        y_true.append(0)
        y_preds.append(s)
        i+=1

    y_true = np.array(y_true)
    y_preds = np.array(y_preds)

    print("Computation Done")
    print("Computing scores")

    auc_score = roc_auc_score(y_true, y_preds)
    print(f"AUC {metric}: ", auc_score)
    
    # plot the roc curve and find best threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=metric)
    plt.savefig(f"{metric}.png")
    plt.clf()
    gmeans = np.sqrt(np.multiply(tpr, (1-fpr)))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    
    y_pred_ind = (y_preds >= thresholds[ix]).astype(int)
    acc_score = accuracy_score(y_true, y_pred_ind)
    pre_score = precision_score(y_true, y_pred_ind)
    rec_score = recall_score(y_true, y_pred_ind)
    print(f"Accuracy {metric}: ", acc_score)
    print(f"Precision {metric}: ", pre_score)
    print(f"Recall {metric}: ", rec_score)


if __name__ == '__main__':
    f = open(sys.argv[1])
    g = json.load(f)

    f.close()

    g = {int(key): [int(v) for v in g[key]] for key in g}

    gFull = nx.DiGraph(g)
    gFull = gFull.to_undirected()
    
    gTrain, gValid, gTest = random_split(gFull)

    link_prediction(gTrain, gValid, metric='jc')
    link_prediction(gTrain, gValid, metric='rai')
    # link_prediction(gTrain, gValid, metric='aai') # can't use, might have zero division
    link_prediction(gTrain, gValid, metric='pa')
    # link_prediction(gTrain, gValid, metric='cnc') # computation too long
