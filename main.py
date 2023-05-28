from src.lib.graphs_general.adjacency_list import AdjacencyList
from src.lib.graphs_general.adjacency_matrix import AdjacencyMatrix

from src.lib.transductive_inference import TransductiveInference

import pandas as pd
import numpy as np
import math

from sklearn.datasets import make_moons

if __name__ == "__main__":
    op = int(input("Options\n\n1 - Create Adjacency List;\n2 - Create Adjacency Matrix;\n0 - Quit.\n--> "))
    if op != 0:
        if op == 1:
            directed = bool(int(input("Will it be directed or non directed? (1/0)\n--> ")))

            v_labeled = bool(int(input("Will it have labeled vertexes? (1/0)\n--> ")))
            v_weighted = bool(int(input("Will it have weighted vertexes? (1/0)\n--> ")))

            e_labeled = bool(int(input("Will it have labeled edges? (1/0)\n--> ")))
            e_weighted = bool(int(input("Will it have weighted edges? (1/0)\n--> ")))

            n = int(input("How many vertexes will it have?\n--> "))

            v_labels = v_weights = e_labels = e_weights = None

            print("Write each value space-separated\n")

            if v_labeled:
                v_labels = tuple(input(f"{n} Vertex labels\n--> ").split(" "))
            if v_weighted:
                v_weights = tuple([float(i) for i in input(f"{n} Vertex weights\n--> ").split(" ")])
            if e_labeled:
                e_labels = tuple(input(f"{n} Edge labels\n--> ").split(" "))
            if e_weighted:
                e_weights = tuple([float(i) for i in input(f"{n} Edge weights\n--> ").split(" ")])

            al = AdjacencyList(n=n, labels=v_labels, weights=v_weights, directed=directed, e_labeled=e_labeled, e_weighted=e_weighted)

            action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n"+
                               "5 - Check adjacency;\n6 - Check if it is empty;\n7 - Check if it is complete (supposing it is simple);\n8 - Print Adjacency List;\n"+
                               "9 - Check amount of vertexes;\n10 - Check amount of edges;\n11 - Print vertex;\n12 - Print edge;\n"+
                               "13 - Print graph info;\n0 - Quit.\n--> "))
            while action != 0:
                if action == 1:
                    # add vertex
                    n = int(input("How many new vertexes?\n--> "))
                
                    labels = weights = None
                    if al.v_weighted():
                        weights = tuple([float(i) for i in input(f"{n} Vertex weights\n--> ").split(" ")])
                    if al.v_labeled():
                        labels = tuple(input(f"{n} Vertex labels\n--> ").split(" "))

                    al.add_vertex(n, labels, weights)
                elif action == 2:
                    # add edge
                    n = int(input("How many new edges?\n--> "))

                    v = []
                    w = []

                    for i in range(n):
                        print(f"Edge {i}\n")

                        tmp_v = tmp_w = None
                        
                        if al.v_labeled():
                            tmp_v = input("From vertex: ")
                            tmp_w = input("To vertex: ")
                        else:
                            tmp_v = int(input("From vertex: "))
                            tmp_w = int(input("To vertex: "))
                        
                        v.append(tmp_v)
                        w.append(tmp_w)

                    labels = weights = None
                    
                    if al.e_weighted():
                        weights = tuple([float(i) for i in input(f"{n} Edge weights\n--> ").split(" ")])
                    if al.e_labeled():
                        labels = tuple(input(f"{n} Edge labels\n--> ").split(" "))
                    
                    al.add_edge(v=tuple(v), w=tuple(w), label=labels, weight=weights)
                elif action == 3:
                    # remove vertex
                    v = None
                    if al.v_labeled():
                        v = input("Vertex to remove: ")
                    else:
                        v = int(input("Vetex to remove: "))
                    al.remove_vertex(v)
                elif action == 4:
                    # remove edge
                    e = v = w = None
                    if bool(int(input("Search by Edge label or Vertexes labels? (1/0)\n--> "))):
                        e = input("Edge to remove: ") if al.e_labeled() else int(input("Edge to remove: "))
                    elif al.v_labeled():
                        v = input("V vertex: ")
                        w = input("W vertex: ")
                    else:
                        v = int(input("V vertex: "))
                        w = int(input("W vertex: "))
                    al.remove_edge(e, v, w)
                elif action == 5:
                    # check if v and w are adjacent
                    v = w = None
                    
                    if al.v_labeled():
                        v = input("V vertex: ")
                        w = input("W vertex: ")
                    else:
                        v = int(input("V vertex: "))
                        w = int(input("W vertex: "))

                    print(f"Is '{v}' and '{w}' adjacent? -->", al.is_adjacent(v, w))
                elif action == 6:
                    # check if graph is empty
                    print("Is the graph empty (0 vertexes)?", al.empty())
                elif action == 7:
                    # check if graph is complete
                    print("Is the graph complete?", al.complete())
                elif action == 8:
                    # print graph
                    verbose = bool(int(input("Verbose print? (1/0)\n--> ")))
                    al.print(verbose=verbose)
                elif action == 9:
                    # check amount of vertexes in the graph
                    print(f"This graph has {al.size()} vertexes.")
                elif action == 10:
                    # check amount of edges in the graph
                    print(f"This graph has {al.edges()} edges.")
                elif action == 11:
                    # print vertex info by label
                    v = input("Vertex label: ") if al.v_labeled() else int(input("Vertex label: "))
                    print(f"Vertex:", al.get_vertex(v)[0].to_string(verbose=True))
                elif action == 12:
                    # print edge info by label
                    e = v = w = None
                    if bool(int(input("Search by Edge label or Vertexes labels? (1/0)\n--> "))):
                        e = input("Edge label: ") if al.e_labeled() else int(input("Edge label: "))
                    elif al.v_labeled():
                        v = input("V vertex: ")
                        w = input("W vertex: ")
                    else:
                        v = int(input("V vertex: "))
                        w = int(input("W vertex: "))

                    print(f"Edge: ", al.get_edge(e, v, w).to_string(verbose=True))
                elif action == 13:
                    # print graph info
                    print("General graph info:", al.info())

                action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n"+
                               "5 - Check adjacency;\n6 - Check if it is empty;\n7 - Check if it is complete (supposing it is simple);\n8 - Print Adjacency List;\n"+
                               "9 - Check amount of vertexes;\n10 - Check amount of edges;\n11 - Print vertex;\n12 - Print edge;\n"+
                               "13 - Print graph info;\n0 - Quit.\n--> "))
            al.print()

am = AdjacencyMatrix(n=5, labels=('a', 'b', 'c', 'd', 'e'), weights=(1.5, 2, 3, 10, 5.3), e_labeled=True, e_weighted=True)

am.add_edge(('a',), ('b',), label=('EdgeA',), weight=(2.5,))
am.print()
am.info()

am.to_gdf("test_gdf_am.gdf")
am.to_csv("test_csv_am.csv")

#data, y = make_moons(500, shuffle=False, noise=0.1, random_state=None)

#ti = TransductiveInference(pd.DataFrame(data))

#ti.fit_predict()
#ti.plot()

al = AdjacencyList(directed=False, e_weighted=True)

al.add_edge((1, 4, 4), (3, 1, 5), weight=(1.5, 5.0, 2.0))
al.print()
print(al.info())

al.to_gdf("test_gdf_al.gdf") # don't exist yet
al.to_csv("test_csv_al.csv")
