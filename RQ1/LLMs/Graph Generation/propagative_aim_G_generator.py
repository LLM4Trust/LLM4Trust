import networkx as nx
import random
import json
import os


def generate_directed_er_graph():
    while True:
        G = nx.DiGraph()
        nodes = list(range(5))
        G.add_nodes_from(nodes)

        edges = []
        for i in range(5):
            for j in range(5):
                if i != j and random.random() < 0.3:  # 30% chance to add an edge
                    weight = random.randint(1, 5)
                    edges.append((i, j, weight))

        G.add_weighted_edges_from(edges)

        # Select two special nodes such that there is a unique path between them and they are not directly connected
        special_nodes = random.sample(nodes, 2)
        src, dst = special_nodes

        if nx.has_path(G, src, dst) and not G.has_edge(src, dst):  # Must not be directly connected
            all_paths = list(nx.all_simple_paths(G, source=src, target=dst))
            if len(all_paths) == 1:  # Ensure only one path exists
                return G, special_nodes


def get_min_path_weight(G, special_nodes):
    src, dst = special_nodes
    path = nx.shortest_path(G, source=src, target=dst, weight='weight')
    min_weight = min(G[u][v]['weight'] for u, v in zip(path, path[1:]))
    return path, min_weight


def save_graph_to_json(G, special_nodes, path, min_weight, filename):
    os.makedirs("save_graphs", exist_ok=True)  # Create the save directory
    filepath = os.path.join("save_graphs", filename)  # Construct the full file path

    graph_data = {
        "edges": [(u, v, G[u][v]['weight']) for u, v in G.edges()],
        "special_nodes": special_nodes,
        "path": path,
        "min_weight": min_weight
    }

    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=4)


# Generate and save multiple graphs
for i in range(100):
    G, special_nodes = generate_directed_er_graph()
    path, min_weight = get_min_path_weight(G, special_nodes)
    save_graph_to_json(G, special_nodes, path, min_weight, f"graph_{i}.json")
    print(f"Graph {i} saved in 'save_graphs/' with special nodes {special_nodes} and min weight {min_weight}")
