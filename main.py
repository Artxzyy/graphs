from src.lib.graphs_general.adjacency_list import AdjacencyList
from src.lib.graphs_general.adjacency_matrix import AdjacencyMatrix

from src.lib.transductive_inference import TransductiveInference

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

#am = AdjacencyMatrix(n=5, labels=('a', 'b', 'c', 'd', 'e'), weights=(1.5, 2, 3, 10, 5.3), e_labeled=True, e_weighted=True)

#am.add_edge(('a',), ('b',), label=('EdgeA',), weight=(2.5,))
#am.print()
#am.info()

#am.to_gdf("test_gdf_am.gdf")
#am.to_csv("test_csv_am.csv")

data, y = make_moons(500, shuffle=True, noise=0.1, random_state=None)

ti = TransductiveInference(data, y)

y_pred = ti.fit_predict()

total = len(y)
corrects = 0
for i in range(total):
    corrects += 1 if y_pred[i] == y[i] else 0

print("corrects:", corrects)
print("total:", total)
print(f"score: {(corrects / total) * 100}%")

ti.plot()
#al = AdjacencyList(directed=False, e_weighted=True)

#al.add_edge((1, 4, 4), (3, 1, 5), weight=(1.5, 5.0, 2.0))
#al.print()
#print(al.info())

#al.to_gdf("test_gdf_al.gdf") # don't exist yet
#al.to_csv("test_csv_al.csv")

#borders = pd.read_csv("tests/StudyCase/contry_borders.csv")




# sample definition



# names = (
#     "bairros_sp", "bairros_fortaleza", "bairros_rj", 
#     "causa_afastamento_1", "causa_afastamento_2", "causa_afastamento_3", 
#     "Motivo_Desligamento", "cbo_ocupacao_2002", "cnae_20_classe", "cnae_95_classe", 
#     "distritos_sp", "vinculo_ativo_31_12", "faixa_etaria", "faixa_hora_contrat", 
#     "faixa_remun_dezem_sm", "faixa_remun_media_sm", "faixa_tempo_emprego", 
#     "escolaridade_apos_2005", "qtd_hora_contr", "idade", "ind_cei_vinculado", 
#     "ind_simples", "mes_admissao", "mes_desligamento", "mun_trab", "municipio", 
#     "nacionalidade", "natureza_juridica", "ind_portador_defic", "qtd_dias_afastamento", 
#     "raca_cor", "regioes_adm_df", "vl_remun_dezembro_nom", "vl_remun_dezembro_sm", 
#     "vl_remun_media_nom", "vl_remun_media_sm", "cnae_20_subclasse", "sexo_trabalhador", 
#     "tamanho_estabelecimento", "tempo_emprego", "tipo_admissao", "tipo_estab1", "tipo_estab2", 
#     "tipo_defic", "tipo_vinculo", "ibge_subsetor", "vl_rem_janeiro_cc", "vl_rem_fevereiro_cc", 
#     "vl_rem_marco_cc", "vl_rem_abril_cc", "vl_rem_maio_cc", "vl_rem_junho_cc", "vl_rem_julho_cc", 
#     "vl_rem_agosto_cc", "vl_rem_setembro_cc", "vl_rem_outubro_cc", "vl_rem_novembro_cc", 
#     "ano_chegada_brasil", "ind_trab_intermitente", "ind_trab_parcial"
# )
# 
# fstream_read = open("tests/StudyCase/amostra_rais_vinc_pub_sp.csv", "r")
# fstream_write = open("tests/StudyCase/rais_random_sample.csv", "w")
# 
# lines = fstream_read.readlines()
# 
# fstream_write.write(f"{';'.join(names)}\n")
# 
# fstream_write.writelines(random.sample(lines, k=int(len(lines) / 10)))
# 
# fstream_read.close()
# 
# fstream_write.flush()
# fstream_write.close()
# 
# # handle data
# 
# df = pd.read_csv("tests/StudyCase/rais_random_sample.csv", sep=";", encoding='ISO-8859-1')
# df_obj = df.select_dtypes(['object'])
# 
# for col in df_obj.columns:
#     # float numbers with dot instead of comma
#     df[col] = df[col].str.replace(',', '.')
#     # remove carriage returns
#     df[col] = df[col].str.replace('\r', "")
#     # remove non-ASCII characters
#     df[col] = df[col].str.encode("ascii", "ignore").str.decode("ascii")
#     # trim columns
#     df[col] = df[col].str.strip()
#     # remove non-ASCII remains
#     df[col] = df[col].str.replace("{ class}", "")
#     df[col] = df[col].str.replace('{', "")
# 
# df.to_csv("tests/StudyCase/rais_random_sample.csv", sep=";", columns=df.columns, index=False)

cols_for_pred = (
    # "bairros_sp", "mun_trab", "municipio", "distritos_sp", 
    "cbo_ocupacao_2002", #"cnae_20_subclasse",# "cnae_20_classe",
    # "tempo_emprego",
    "escolaridade_apos_2005", "qtd_hora_contr",
    "tipo_defic",
    "raca_cor", "sexo_trabalhador", "idade", 
    "tamanho_estabelecimento",
    "faixa_remun_media_sm"
)

df = pd.read_csv("tests/StudyCase/rais_random_sample.csv", sep=";")

cols_to_drop = []
for col in df.columns:
    if col not in cols_for_pred:
        cols_to_drop.append(col)

df = df.drop(cols_to_drop, axis=1)
df["cbo_ocupacao_2002"] = df["cbo_ocupacao_2002"].fillna("0")
df["cbo_ocupacao_2002"] = df["cbo_ocupacao_2002"].str.replace("-", "")
df = df[df["cbo_ocupacao_2002"].str.startswith(("2",))]

df = df.fillna(0)

#df = df.drop([i for i in df.index[int(len(df.index) / 10) : len(df.index)]])
df = df[df["faixa_remun_media_sm"] != 99]


for col in df.select_dtypes(['object']).columns:
    # df[col] = df[col].str.replace('-', "")
    df[col] = pd.to_numeric(df[col])

ti = TransductiveInference(df, df["faixa_remun_media_sm"], y_name="faixa_remun_media_sm")

y_pred = ti.fit_predict()
y_true = [i for i in df["faixa_remun_media_sm"].values]


print("Y_TRUE:", y_true)
print("\n\n\n\n\n\n")
print("Y_PRED:", y_pred)

total = len(df)
corrects = 0
for i in range(total):
    corrects += 1 if y_pred[i] == y_true[i] else 0

print("corrects:", corrects)
print("total:", total)
print(f"score: {(corrects / total) * 100}%")

cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(0, 12)])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(0, 12)])

disp.plot()
plt.show()