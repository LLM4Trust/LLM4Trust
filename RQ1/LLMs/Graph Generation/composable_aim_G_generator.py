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

        special_nodes = random.sample(nodes, 2)
        src, dst = special_nodes
        if not G.has_edge(src, dst):  # Ensure there is no direct edge from src to dst
            all_paths = list(nx.all_simple_paths(G, src, dst))
            if len(all_paths) > 1:
                all_paths = list(nx.all_simple_paths(G, source=src, target=dst))

                min_path = None
                max_path = None
                min_path_weight = float('inf')
                max_path_weight = float('-inf')

                for path in all_paths:
                    path_weight = min(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                    if path_weight < min_path_weight:
                        min_path_weight = path_weight
                        min_path = path
                    if path_weight > max_path_weight:
                        max_path_weight = path_weight
                        max_path = path

                if min_path != max_path:
                    return G, special_nodes, min_path_weight, max_path_weight, min_path, max_path


def save_graph_to_json(G, special_nodes, min_path_weight, max_path_weight, min_path, max_path, filename):
    os.makedirs("save_graphs", exist_ok=True)  # Create output directory if it doesn't exist
    filepath = os.path.join("save_graphs", filename)  # Construct full file path

    graph_data = {
        "edges": [(u, v, G[u][v]['weight']) for u, v in G.edges()],
        "special_nodes": special_nodes,
        "num_min": min_path_weight,
        "num_max": max_path_weight,
        "path_min": min_path,
        "path_max": max_path
    }
    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=4)


# Generate and save multiple graphs
for i in range(100):
    G, special_nodes, min_path_weight, max_path_weight, min_path, max_path = generate_directed_er_graph()
    save_graph_to_json(G, special_nodes, min_path_weight, max_path_weight, min_path, max_path, f"graph_{i}.json")
    print(f"Graph {i} saved in 'save_graphs/' with special nodes {special_nodes}")
