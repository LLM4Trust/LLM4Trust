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
        a, b = special_nodes
        if G.has_edge(a, b) and not G.has_edge(b, a):
            if not nx.has_path(G, b, a):  # Ensure there is no path from b to a
                special_nodes = b, a
                return G, special_nodes  # Return the graph and the special nodes a, b


def save_graph_to_json(G, special_nodes, filename):
    os.makedirs("save_graphs", exist_ok=True)  # Create the output directory if it doesn't exist
    filepath = os.path.join("save_graphs", filename)  # Construct the file path

    graph_data = {
        "edges": [(u, v, G[u][v]['weight']) for u, v in G.edges()],
        "special_nodes": special_nodes,
        "answer": 0
    }

    with open(filepath, "w") as f:
        json.dump(graph_data, f, indent=4)


# Generate and save multiple graphs
for i in range(100):
    G, special_nodes = generate_directed_er_graph()
    save_graph_to_json(G, special_nodes, f"graph_{i}.json")
    print(f"Graph {i} saved in 'save_graphs/' with special nodes {special_nodes} and weight {0}")
