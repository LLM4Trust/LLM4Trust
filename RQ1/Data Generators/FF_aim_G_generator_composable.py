import networkx as nx
import random
import json
import os
from collections import deque


def forest_fire_graph_standard(num_nodes=10, p_forward=0.30, p_backward=0.20, weight_max=5):
    """
    Generate a directed graph using the standard Forest Fire model,
    including forward and backward burning, based on Leskovec et al. (2005).

    Parameters:
        num_nodes (int): Number of nodes (default is 10)
        p_forward (float): Forward burning probability (default is 0.30)
        p_backward (float): Backward burning probability (default is 0.20)
        weight_max (int): Maximum edge weight (range is [1, weight_max], default is 5)

    Returns:
        nx.DiGraph: A directed graph with edge weights
    """
    G = nx.DiGraph()
    G.add_node(0)  # Initialize the graph starting from node 0

    for new_node in range(1, num_nodes):
        G.add_node(new_node)

        # Randomly select one or more existing nodes as ignition seeds
        seeds = [random.randint(0, new_node - 1)]  # Select only from existing nodes
        visited = set()
        queue = deque(seeds)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Add edge: new_node -> current (new node points to old node)
            if not G.has_edge(new_node, current):
                weight = random.randint(1, weight_max)
                G.add_edge(new_node, current, weight=weight)

            # Forward burning: out-neighbors of the current node
            for neighbor in G.successors(current):
                if random.random() < p_forward and neighbor not in visited:
                    queue.append(neighbor)

            # Backward burning: in-neighbors of the current node
            for neighbor in G.predecessors(current):
                if random.random() < p_backward and neighbor not in visited:
                    queue.append(neighbor)

    return G


def generate_directed_ff_graph():
    while True:
        G = forest_fire_graph_standard(num_nodes=5, p_forward=0.35, p_backward=0.25, weight_max=5)
        nodes = list(range(5))

        special_nodes = random.sample(nodes, 2)
        src, dst = special_nodes
        if not G.has_edge(src, dst):  # Ensure src does not directly connect to dst
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
    os.makedirs("save_graphs_composable", exist_ok=True)  # Create output directory
    filepath = os.path.join("save_graphs_composable", filename)  # Construct full path

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
    G, special_nodes, min_path_weight, max_path_weight, min_path, max_path = generate_directed_ff_graph()
    save_graph_to_json(G, special_nodes, min_path_weight, max_path_weight, min_path, max_path, f"graph_{i}.json")
    print(f"Graph {i} saved in 'save_graphs_composable/' with special nodes {special_nodes}")
