import pandas as pd
import networkx as nx


def build_graph_from_edges(edges):
    """
    Build a directed graph from the given edge list, where each edge has 'label' and 'timestamp' attributes.
    """
    G = nx.DiGraph()

    # Iterate through each row and add the edge to the graph
    for _, row in edges.iterrows():
        G.add_edge(
            row['src'],  # source node
            row['dst'],  # target node
            label=row['label'],  # edge label
            timestamp=row['timestamp']  # edge timestamp
        )

    return G


def get_max_timestamp(path, G):
    # Get the maximum timestamp among all edges in the path
    max_timestamp = max(G[u][v]['timestamp'] for u, v in zip(path[:-1], path[1:]))
    return max_timestamp


def subgraph_generation(train_file, src, dst, max_1, hop, max_2):
    if isinstance(train_file, str):
        train_df = pd.read_csv(train_file)
        # Build graph from training data
        G_train = build_graph_from_edges(train_df)
    else:
        G_train = build_graph_from_edges(train_file)
    tag = 0
    # return tag flag graph
    if src in G_train and dst in G_train:
        # Case 1: both nodes exist and a path exists
        if nx.has_path(G_train, src, dst):
            subgraph_edges = set()
            for cutoff in range(1, hop + 1):
                all_paths = []
                paths = list(nx.all_simple_paths(G_train, source=src, target=dst, cutoff=cutoff))
                if not paths:
                    continue
                for path in paths:
                    max_timestamp = get_max_timestamp(path, G_train)
                    all_paths.append((path, max_timestamp))
                all_paths.sort(key=lambda x: x[1], reverse=True)

                for path, _ in all_paths:
                    path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
                    if len(subgraph_edges) + len(set(path_edges) - subgraph_edges) > max_1:
                        break
                    subgraph_edges.update(path_edges)
            if not subgraph_edges:
                return 1, 0, -1
            subgraph = G_train.edge_subgraph(subgraph_edges)
            print(f"Test edge ({src}, {dst}) falls into Case 1: Path exists, found paths up to depth {hop}, subgraph: {subgraph}")
            tag = 1
            return tag, 1, subgraph
        else:
            return 1, 0, -1
    else:
        if src in G_train:
            outgoing_edges = []
            for u, v, data in G_train.edges(src, data=True):
                outgoing_edges.append((u, v, data['label'], data['timestamp']))
            outgoing_edges.sort(key=lambda x: x[3], reverse=True)
            selected_edges = []
            for edge in outgoing_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 2: src exists, subgraph with 1-hop and 2-hop neighbors: {one_order_subgraph}")
                    tag = 2
                    return tag, 1, one_order_subgraph
                selected_edges.append(edge)

            second_order_edges = []
            for _, neighbor, _, _ in selected_edges:
                for u, v, data in G_train.edges(neighbor, data=True):
                    second_order_edges.append((u, v, data['label'], data['timestamp']))
            second_order_edges.sort(key=lambda x: x[3], reverse=True)
            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 2: src exists, subgraph with 1-hop and 2-hop neighbors: {one_order_subgraph}")
                    tag = 2
                    return tag, 1, one_order_subgraph
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) falls into Case 2: src exists, subgraph with 1-hop and 2-hop neighbors: {one_order_subgraph}")
            tag = 2
            return tag, 1, one_order_subgraph

        elif dst in G_train:
            incoming_edges = []
            for u, v, data in G_train.in_edges(dst, data=True):
                incoming_edges.append((u, v, data['label'], data['timestamp']))
            incoming_edges.sort(key=lambda x: x[3], reverse=True)

            selected_edges = []
            for edge in incoming_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 3: dst exists, subgraph with 1-hop and 2-hop in-edges: {one_order_subgraph}")
                    tag = 3
                    return tag, 1, one_order_subgraph
                selected_edges.append(edge)

            second_order_edges = []
            for u, _, _, _ in selected_edges:
                for src2, mid, data in G_train.in_edges(u, data=True):
                    second_order_edges.append((src2, mid, data['label'], data['timestamp']))

            second_order_edges.sort(key=lambda x: x[3], reverse=True)

            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 3: dst exists, subgraph with 1-hop and 2-hop in-edges: {one_order_subgraph}")
                    tag = 3
                    return tag, 1, one_order_subgraph
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) falls into Case 3: dst exists, subgraph with 1-hop and 2-hop in-edges: {one_order_subgraph}")
            tag = 3
            return tag, 1, one_order_subgraph

        else:
            if isinstance(train_file, str):
                train_df = pd.read_csv(train_file)
            else:
                train_df = train_file
                if len(train_df) < max_2:
                    return 4, 0, -1
            last_edges = train_df.tail(max_2)
            G_last = build_graph_from_edges(last_edges)
            print(f"Test edge ({src}, {dst}) falls into Case 4: nodes do not exist, use last {max_2} edges of the training set to build the subgraph: {G_last}")
            tag = 4
            return tag, 1, G_last


if __name__ == '__main__':
    train_file = "graph/train_edges.csv"
    train_df = pd.read_csv(train_file)

    half_index = len(train_df) // 2
    candidate_edges = train_df.iloc[half_index:].copy()

    max_1 = 70
    hop = 3
    max_2 = 70

    output_file = "train_tag4.csv"
    pd.DataFrame(
        columns=['test_src', 'test_dst', 'test_label', 'test_timestamp', 'tag', 'flag', 'subgraph_edges']).to_csv(
        output_file, index=False)

    tag4_count = 0
    candidate_edges = candidate_edges.sample(frac=1).reset_index(drop=True)
    candidate_indices = list(range(len(candidate_edges)))

    for idx in candidate_indices:
        row = candidate_edges.iloc[idx]
        src = int(row['src'])
        dst = int(row['dst'])
        label = int(row['label'])

        if label == 0:
            continue

        current_train = train_df.iloc[:half_index + idx]
        G_4 = build_graph_from_edges(current_train)

        if src in G_4 and dst in G_4:
            tag = 1
            print('tag1')
            continue
        elif src in G_4:
            tag = 2
            print('tag2')
            continue
        elif dst in G_4:
            tag = 3
            print('tag3')
            continue
        else:
            tag = 4

        tag, flag, graph = subgraph_generation(current_train, src, dst, max_1, hop, max_2)

        if tag == 4 and tag4_count < 100:
            graph_edges = list()
            if graph is not None and graph != -1:
                for u, v, data in graph.edges(data=True):
                    graph_edges.append((int(u), int(v), int(data['label']), float(data['timestamp'])))

            current_data = pd.DataFrame([{
                'src': int(src),
                'dst': int(dst),
                'label': int(row['label']) if 'label' in row else None,
                'timestamp': float(row['timestamp']) if 'timestamp' in row else None,
                'tag': tag,
                'flag': flag,
                'graph': list(graph_edges)
            }])

            current_data.to_csv(output_file, mode='a', header=False, index=False)

            tag4_count += 1
            print(f"Found {tag4_count} tag4 edges")

            if tag4_count >= 100:
                break

    print(f"Processing complete. Results saved to {output_file}, with {tag4_count} tag4 edges.")
