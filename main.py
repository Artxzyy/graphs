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

# data: pd.Series = pd.read_csv("tests/basic_csv.csv", sep=",").squeeze(axis=0)
# 
# print(data)
# print()
# 
# ti = TransductiveInference(data)
# 
# afm = ti.create_affinity_matrix()
# print(afm)
# 
# print("\n\n")
# am = AdjacencyMatrix()
# 
# np_array = am.get_np_array()
# 
# print(np_array)
# 
# np.fill_diagonal(a=np_array, val=0.0)
# 
# def m_values(xi, xj):
#     sigma2 = ti.std() ** 2
#     den = ((2 * math.pi * sigma2) ** (1/2))
#     exp_den = 2 * sigma2
#     exp_total = (((xi - xj) ** 2)) / exp_den
#     # return (1 / den) * math.exp(-((xi - xj) ** (2)) / (2 * (ti.std() ** (2))))
#     return (1 / den) * (math.exp( - exp_total))
# 
# print(len(np_array[0]))
# 
# for i in range(len(np_array[0])):
#     for j in range(len(np_array[0])):
#         np_array[i][j] = m_values(np_array[i][i], np_array[j][j])
# 
# print(np_array)

print("\n\n------------------------------------------------------------------------\n\n")

am = AdjacencyMatrix(n=2, directed=True, e_weighted=True)

am.print()

print("\n\n 1 \n\n")

am.add_vertex(1)

am.print()

print("\n\n 2 \n\n")

am.add_edge((1,), (2,), weight=(5,))

am.print()

print("\n\n 3 \n\n")

am.update_edge(e=1, new_weight=10)

am.print()

print("\n\n 4 \n\n")

am.remove_edge(1)

am.print()

print("\n\n 5 \n\n")

am.remove_vertex(1)

am.print()

print("\n\n-----------\n\n")

data, y = make_moons(500, shuffle=True, noise=0.1, random_state=None)

ti = TransductiveInference(pd.DataFrame(data))

print("\nAffinity Matrix\n")

print(ti.affinity_matrix)

print("\nS {D^(-1/2)WD^(-1/2)}\n")

print(ti.s)

print("\nY",ti.y_input())

ti.fit()
