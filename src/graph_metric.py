"""
To check the metrics of a graph, do 
```python graph_metric.py graph.json```
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

def plot_distribution(graph):
    cite_num = {}

    for paper in graph.keys():
        # print(paper, graph[paper])
        num = len(graph[paper])

        if num in cite_num:
            cite_num[num] += 1
        else:
            cite_num[num] = 1
    
    arr = sorted(list(cite_num.items()))
    
    arr = np.array(arr)
    
    plt.bar(arr[:, 0], arr[:, 1])

    plt.yscale('log')
    plt.xlabel('# of citations')
    plt.ylabel('# of papers (Log scale)')

    plt.savefig('distribution.png')


def count_cc(graph):
    """
        Count connected components in an undirected graph
    """
    iso_vertex = 0
    cc = 0
    trimmed_graph = copy.deepcopy(graph)
    for v in graph:
        if len(graph[v]) == 0:
            cc += 1
            iso_vertex += 1
            trimmed_graph.pop(v)
    
    status = {key: 0 for key in trimmed_graph}

    unvisited_nodes = set([key for key in trimmed_graph])
    
    while len(unvisited_nodes) != 0:
        cc += 1
        root = tuple(unvisited_nodes)[0]

        assert status[root] == 0

        nodes_to_visit = [root]
    
        while len(nodes_to_visit) != 0:
            c = nodes_to_visit[-1]
            
            nodes_to_visit.pop() # pop the node
            if status[c] == 1:
                continue

            status[c] = 1
            unvisited_nodes.remove(c)

            for n in graph[c]:
                if status[n] == 0:
                    nodes_to_visit.append(n)

    return cc, iso_vertex

def detect_cycle(graph, verbose=0):
    global counter, self_loops

    status = {key: 0 for key in graph}

    for c in graph:
        if status[c] == 0:
            dfs_visit(graph, status, c, verbose)
                
    return True

def dfs_visit(graph, status, c, verbose=0):
    global counter, self_loops

    status[c] = 1
    neighbors = graph[c]
    
    for n in neighbors:
        if status[n] == 0:
            dfs_visit(graph, status, n, verbose)
           
        elif status[n] == 1:
            if verbose == 1:
                print("Cycle detected: {} and {}".format(c, n))
                print(c, graph[c])
                print(n, graph[n])

            counter += 1
            if c == n:
                self_loops += 1
            

    status[c] = 2

def make_undirected(graph):
    """
        Convert a directed graph to an undirected graph
    """
    un_graph = copy.deepcopy(graph)

    for v in graph.keys():
        for n in graph[v]:
            un_graph[n].append(v)
    
    return un_graph


if __name__ == '__main__':
    f = open(sys.argv[1])

    graph = json.load(f)

    print(len(graph))
    
    # plot the bar chart distribution
    plot_distribution(graph)


    # count the number of cycles
    counter = 0
    self_loops = 0

    detect_cycle(graph, verbose=0)

    print("Number of cycles (include self loops):", counter)
    print("Number of self-loops:", self_loops)

    # Count the number of connected components (Treat the graph as undirected)
    un_graph = make_undirected(graph)

    cc, iso_vertex = count_cc(un_graph)
    print("Connected components: {}".format(cc))
    print("Connected components (exclude iso vertrices): {}".format(cc - iso_vertex))
    print("Isolated vertex: {}".format(iso_vertex))



    
    

    
