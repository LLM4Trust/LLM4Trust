import networkx as nx
import random
import json
import os


def generate_directed_sb_graph():
    while True:
        G = nx.DiGraph()
        num_nodes = 5
        nodes = list(range(5))

        sizes = [2, 3]  # Two communities
        blocks = [0] * 2 + [1] * 3

        # Add nodes and assign community labels
        for i in range(num_nodes):
            G.add_node(i, block=blocks[i])

        # Define connection probabilities
        in_class_p = 0.4
        cross_class_p = 0.2

        # Iterate over all possible directed edges
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # Skip self-loops
                same_class = blocks[i] == blocks[j]
                prob = in_class_p if same_class else cross_class_p
                if random.random() < prob:
                    G.add_edge(i, j, weight=random.randint(1, 5))

        # Select two special nodes such that they are not directly connected
        special_nodes = random.sample(nodes, 2)
        src, dst = special_nodes
        if not G.has_edge(src, dst):  # Ensure src does not directly point to dst
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
    os.makedirs("save_graphs_composable", exist_ok=True)  # Create the save directory
    filepath = os.path.join("save_graphs_composable", filename)  # Construct the full file path

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
    G, special_nodes, min_path_weight, max_path_weight, min_path, max_path = generate_directed_sb_graph()
    save_graph_to_json(G, special_nodes, min_path_weight, max_path_weight, min_path, max_path, f"graph_{i}.json")
    print(f"Graph {i} saved in 'save_graphs_composable/' with special nodes {special_nodes}")
