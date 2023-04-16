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

            al = AdjacencyList(n=n, labels=v_labels, weights=v_weights)

            action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n\
                               5 - Check adjacency;\n6 - Check if it is empty;\n7 - Print Adjacency List;\n0 - Quit.\n--> "))
            while action != 0:
                if action == 1:
                    n = int(input("How many new vertexes?\n--> "))
                
                    labels = weights = None
                    if v_weighted:
                        weights = tuple(input(f"{n} Vertex labels\n--> ").split(" "))
                    if v_labels:
                        labels = tuple(input(f"{n} Vertex labels\n--> ").split(" "))
                    al.add_vertex(n, labels, weights)
                elif action == 2:
                    v = input("From vertex: ") if v_labeled else int(input("From vertex: "))
                    w = input("To vertex: ") if v_labeled else int(input("To vertex: "))
                    
                    e_label = e_weight = None

                    if e_labeled:
                        e_label = input("Edge label: ")
                    if e_weighted:
                        e_weight = input("Edge weight: ")
                    
                    al.add_edge(v=v, w=w, label=e_label, weight=e_weight)
                elif action == 3:
                    pass
                elif action == 4:
                    pass
                elif action == 5:
                    v = input("V vertex: ") if v_labeled else int(input("V vertex: "))
                    w = input("W vertex: ") if v_labeled else int(input("W vertex: "))

                    print(f"Is '{v}' and '{w}' adjacent? -->", al.is_adjacent(v, w))


                action = int(input("Select an option\n\n1 - Add vertex;\n2 - Add edge;\n3 - Remove vertex;\n4 - Remove Edge;\n\
                               5 - Check adjacency;\n6 - Check if it is empty;\n7 - Print Adjacency List;\n0 - Quit.\n--> "))
            al.print()


