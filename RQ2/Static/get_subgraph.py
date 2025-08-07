import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./advogato/advogato.csv')  # use './advogato/advogato.csv' or './pgp/pgp.csv!!!
labels = df['label'].to_numpy()
edges = df.iloc[:, :2].to_numpy()
hop = 3
path_length = 60  # 60 for advogato and 5 for pgp!!!

train_edges, test_edges, y_train, y_test = train_test_split(edges, labels, test_size=0.2, random_state=42)
print(len(y_test))

# Step 1: Build a directed graph from the training set
G = nx.DiGraph()
# Also store the label as an edge attribute
for i in range(len(train_edges)):
    G.add_edge(train_edges[i][0], train_edges[i][1], label=y_train[i])

# Step 2: Find all simple paths
flag1, flag2 = 0, 0
for i, test_edge in enumerate(test_edges):  # use test_edges or train_edges!!!
    if test_edge[0] not in G.nodes() or test_edge[1] not in G.nodes():  # Skip if nodes are not in the training set
        flag1 += 1
        continue

    path_list = []
    for cutoff in range(2, hop + 1):  # Start from 2 to exclude direct trustor-trustee links
        forward_paths = list(nx.all_simple_paths(G, source=test_edge[0], target=test_edge[1], cutoff=cutoff))
        if len(forward_paths) == 0:  # No paths found
            flag2 += 1
            continue

        for path in forward_paths:
            edges = list(zip(path[:-1], path[1:]))
            if len(edges) < cutoff:  # Skip incomplete paths
                continue
            if len(path_list) + len(edges) > path_length:  # Stop if the path list exceeds the limit
                break

            for u, v in edges:
                if u == test_edge[0] and v == test_edge[1]:  # Skip direct trustor-trustee edges
                    continue
                edge_data = G.get_edge_data(u, v)
                label = edge_data['label']
                path_list.append((u, v, label))

    if len(path_list) > 0:
        # Get the ground-truth label for the test edge
        true_label = y_test[i]
        # true_label = y_train[i]  # use y_test or y_train!!!

        with open('./advogato/test_paths{}_hop{}.txt'.format(path_length, hop), 'a') as f:  # use test_paths or train_paths!!!
            f.write(f"{(test_edge[0], test_edge[1])}\t{true_label}\t{path_list}\n")

print(flag1, flag2)
