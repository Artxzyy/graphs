from lib.adjacency_list import AdjacencyList

if __name__ == "__main__":
    op = int(input("Options\n\n1 - Create Adjacency List;\n2 - Create Adjacency Matrix;\n0 - Quit.\n--> "))
    if op != 0:
        if op == 1:
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

            al = AdjacencyList(n=n, labels=v_labels, weights=v_weights, e_labeled=e_labeled, e_weighted=e_weighted)

            action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n"+
                               "5 - Check adjacency;\n6 - Check if it is empty;\n7 - Print Adjacency List;\n0 - Quit.\n--> "))
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
                    
                    e = None
                    if al.e_labeled():
                        e = input("Edge to remove: ")
                    else:
                        e = int(input("Edge to remove: "))
                    al.remove_edge(e)
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
                    # print graph
                    
                    al.print()

                action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n"+
                               "5 - Check adjacency;\n6 - Check if it is empty;\n7 - Print Adjacency List;\n0 - Quit.\n--> "))
            al.print()
