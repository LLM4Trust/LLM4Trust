import pandas as pd
import networkx as nx
import json
import ast
import argparse

def save_file(output_file, content):
    # Construct the filename, optionally including model and task
    filename = output_file
    # Open the file and write the content
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, indent=4))  # Convert content to formatted JSON string
        f.write("\n")  # Add newline


def build_graph_from_edges(edges):
    """
    Build a directed multigraph from a given list of edges, each with 'label' and 'timestamp' attributes.
    """
    G = nx.DiGraph()

    # Iterate through each edge and add it to the graph
    for _, row in edges.iterrows():
        # Assume the edge has 'label' and 'timestamp' attributes
        G.add_edge(
            row['src'],      # Source node
            row['dst'],      # Destination node
            label=row['label'],        # Edge label
            timestamp=row['timestamp']  # Edge timestamp
        )

    return G


def get_max_timestamp(path, G):
    # Get the maximum timestamp among all edges in the path
    max_timestamp = max(G[u][v]['timestamp'] for u, v in zip(path[:-1], path[1:]))
    return max_timestamp


def subgraph_generation(train_file, src, dst, max_1, hop, max_2):
    if isinstance(train_file, str):
        train_df = pd.read_csv(train_file)
        # Build the graph from the training set
        G_train = build_graph_from_edges(train_df)
    else:
        G_train = build_graph_from_edges(train_file)

    if src in G_train and dst in G_train:
        # Case 1: Both nodes exist and a path exists between them
        if nx.has_path(G_train, src, dst):
            # Find all paths from depth 1 to 'hop'
            subgraph_edges = set()
            for cutoff in range(1, hop+1):
                all_paths = []
                paths = list(nx.all_simple_paths(G_train, source=src, target=dst, cutoff=cutoff))
                if not paths:
                    continue
                # Compute the maximum timestamp of each path
                for path in paths:
                    max_timestamp = get_max_timestamp(path, G_train)
                    all_paths.append((path, max_timestamp))
                all_paths.sort(key=lambda x: x[1], reverse=True)

                for path, _ in all_paths:
                    path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
                    # Check whether adding this path exceeds max_1
                    if len(subgraph_edges) + len(set(path_edges) - subgraph_edges) > max_1:
                        break
                    subgraph_edges.update(path_edges)
            if not subgraph_edges:
                return -1
            # Create a subgraph using only the selected path edges
            subgraph = G_train.edge_subgraph(subgraph_edges)
            print(f"Test edge ({src}, {dst}) falls into Case 1: Path exists, subgraph with hop={hop} found: {subgraph}")
            return subgraph
        else:
            return -1
    else:
        # One or both nodes do not exist
        if src in G_train:
            # Case 2: Only the src node exists
            # Get all outgoing edges from src and sort by timestamp (descending)
            outgoing_edges = []
            for u, v, data in G_train.edges(src, data=True):
                outgoing_edges.append((u, v, data['label'], data['timestamp']))
            outgoing_edges.sort(key=lambda x: x[3], reverse=True)
            selected_edges = []
            for edge in outgoing_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 2: Only src exists, subgraph using 1st and 2nd-order neighbors: {one_order_subgraph}")
                    return one_order_subgraph
                selected_edges.append(edge)

            # Retrieve second-order outgoing edges
            second_order_edges = []
            for _, neighbor, _, _ in selected_edges:
                for u, v, data in G_train.edges(neighbor, data=True):
                    second_order_edges.append((u, v, data['label'], data['timestamp']))
            second_order_edges.sort(key=lambda x: x[3], reverse=True)

            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 2: Only src exists, subgraph using 1st and 2nd-order neighbors: {one_order_subgraph}")
                    return one_order_subgraph
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) falls into Case 2: Only src exists, subgraph using 1st and 2nd-order neighbors: {one_order_subgraph}")
            return one_order_subgraph

        elif dst in G_train:
            # Case 3: Only the dst node exists
            # Get all incoming edges to dst
            incoming_edges = []
            for u, v, data in G_train.in_edges(dst, data=True):
                incoming_edges.append((u, v, data['label'], data['timestamp']))
            incoming_edges.sort(key=lambda x: x[3], reverse=True)

            selected_edges = []
            for edge in incoming_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 3: Only dst exists, subgraph using 1st and 2nd-order in-edges: {one_order_subgraph}")
                    return one_order_subgraph
                selected_edges.append(edge)

            # Retrieve second-order incoming edges
            second_order_edges = []
            for u, _, _, _ in selected_edges:
                for src2, mid, data in G_train.in_edges(u, data=True):
                    second_order_edges.append((src2, mid, data['label'], data['timestamp']))
            second_order_edges.sort(key=lambda x: x[3], reverse=True)

            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) falls into Case 3: Only dst exists, subgraph using 1st and 2nd-order in-edges: {one_order_subgraph}")
                    return one_order_subgraph
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) falls into Case 3: Only dst exists, subgraph using 1st and 2nd-order in-edges: {one_order_subgraph}")
            return one_order_subgraph

        else:
            # Case 4: Neither node exists, use the last max_2 edges from the training set
            if isinstance(train_file, str):
                train_df = pd.read_csv(train_file)
            else:
                train_df = train_file
                if len(train_df) < max_2:
                    return -1
            last_edges = train_df.tail(max_2)
            G_last = build_graph_from_edges(last_edges)
            print(f"Test edge ({src}, {dst}) falls into Case 4: Both nodes missing, subgraph built from the last {max_2} edges in training set: {G_last}")
            return G_last


def shot_subgraph_generation(train_file, src, dst, max_1, hop, max_2):
    G_train = build_graph_from_edges(train_file)
    tag = 0

    if src in G_train and dst in G_train:
        # Case 1: Both nodes exist and there is a path
        if nx.has_path(G_train, src, dst):
            # Find all paths from src to dst with depth from 1 to 'hop'
            subgraph_edges = set()
            for cutoff in range(1, hop+1):
                all_paths = []
                paths = list(nx.all_simple_paths(G_train, source=src, target=dst, cutoff=cutoff))
                if not paths:
                    continue
                # Calculate the maximum timestamp of each path
                for path in paths:
                    max_timestamp = get_max_timestamp(path, G_train)
                    all_paths.append((path, max_timestamp))
                all_paths.sort(key=lambda x: x[1], reverse=True)

                for path, _ in all_paths:
                    path_edges = [(u, v) for u, v in zip(path[:-1], path[1:])]
                    # Check whether adding this path would exceed max_1
                    if len(subgraph_edges) + len(set(path_edges) - subgraph_edges) > max_1:
                        break
                    subgraph_edges.update(path_edges)
            if not subgraph_edges:
                return -1, tag
            # Create subgraph using only edges in the paths
            subgraph = G_train.edge_subgraph(subgraph_edges)
            print(f"Test edge ({src}, {dst}) is Case 1: Path exists, found paths with depth up to {hop}, subgraph: {subgraph}")
            tag = 1
            return subgraph, tag
        else:
            return -1, tag
    else:
        # One or both nodes do not exist
        if src in G_train:
            # Get all outgoing edges from src and sort by timestamp in descending order
            outgoing_edges = []
            for u, v, data in G_train.edges(src, data=True):
                # 'data' is a dictionary containing edge attributes (e.g., label and timestamp)
                outgoing_edges.append((u, v, data['label'], data['timestamp']))
            outgoing_edges.sort(key=lambda x: x[3], reverse=True)

            selected_edges = []
            for edge in outgoing_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) is Case 2: Only src exists, using 1st- and 2nd-order neighbors, subgraph: {one_order_subgraph}")
                    tag = 2
                    return one_order_subgraph, tag
                selected_edges.append(edge)

            # Get outgoing edges of the target nodes of the first-order neighbors
            second_order_edges = []
            for _, neighbor, _, _ in selected_edges:
                for u, v, data in G_train.edges(neighbor, data=True):
                    second_order_edges.append((u, v, data['label'], data['timestamp']))
            second_order_edges.sort(key=lambda x: x[3], reverse=True)

            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) is Case 2: Only src exists, using 1st- and 2nd-order neighbors, subgraph: {one_order_subgraph}")
                    tag = 2
                    return one_order_subgraph, tag
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) is Case 2: Only src exists, using 1st- and 2nd-order neighbors, subgraph: {one_order_subgraph}")
            tag = 2
            return one_order_subgraph, tag

        elif dst in G_train:
            # Get all incoming edges to dst and sort by timestamp
            incoming_edges = []
            for u, v, data in G_train.in_edges(dst, data=True):
                incoming_edges.append((u, v, data['label'], data['timestamp']))
            incoming_edges.sort(key=lambda x: x[3], reverse=True)

            selected_edges = []
            for edge in incoming_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) is Case 3: Only dst exists, using 1st- and 2nd-order in-edges, subgraph: {one_order_subgraph}")
                    tag = 3
                    return one_order_subgraph, tag
                selected_edges.append(edge)

            # Get incoming edges of the source nodes of first-order neighbors
            second_order_edges = []
            for u, _, _, _ in selected_edges:
                for src2, mid, data in G_train.in_edges(u, data=True):
                    second_order_edges.append((src2, mid, data['label'], data['timestamp']))
            second_order_edges.sort(key=lambda x: x[3], reverse=True)

            for edge in second_order_edges:
                if len(selected_edges) >= max_2:
                    one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
                    print(f"Test edge ({src}, {dst}) is Case 3: Only dst exists, using 1st- and 2nd-order in-edges, subgraph: {one_order_subgraph}")
                    tag = 3
                    return one_order_subgraph, tag
                selected_edges.append(edge)

            one_order_subgraph = G_train.edge_subgraph([(u, v) for u, v, _, _ in selected_edges])
            print(f"Test edge ({src}, {dst}) is Case 3: Only dst exists, using 1st- and 2nd-order in-edges, subgraph: {one_order_subgraph}")
            tag = 3
            return one_order_subgraph, tag

        else:
            # Case 4: Both nodes are absent, use the last max_2 edges from the training set
            if isinstance(train_file, str):
                train_df = pd.read_csv(train_file)
            else:
                train_df = train_file
                if len(train_df) < max_2:
                    return -1, tag
            last_edges = train_df.tail(max_2)
            G_last = build_graph_from_edges(last_edges)
            print(f"Test edge ({src}, {dst}) is Case 4: Both nodes missing, using the last {max_2} edges from training set, subgraph: {G_last}")
            tag = 4
            return G_last, tag


def build_few_shot_examples(args, train_csv_path, src, dst, flag):
    df = pd.read_csv(train_csv_path)
    G_train = build_graph_from_edges(df)
    flag = flag
    # Check required columns
    if not all(col in df.columns for col in ['src', 'dst', 'label', 'timestamp']):
        raise ValueError("Missing required columns in the CSV file: src, dst, label, timestamp")

    ordinal_words = ["first", "second", "third", "fourth", "fifth", "sixth"]
    prompt = ""
    
    if flag == 4:
        df_4 = pd.read_csv('train_tag4.csv')

        # Check required columns for tag-4 examples
        required_columns = ['src', 'dst', 'label', 'timestamp', 'tag', 'flag', 'graph']
        if not all(col in df_4.columns for col in required_columns):
            raise ValueError(f"Missing required columns in the CSV file: {', '.join(required_columns)}")

        # Randomly select 3 samples
        samples = df_4.sample(n=3).reset_index()

        for i, row in samples.iterrows():
            src = int(row['src'])
            dst = int(row['dst'])
            label = int(row['label'])
            timestamp = row['timestamp']
            tag = int(row['tag'])

            try:
                edge_list = ast.literal_eval(row['graph'])  # Parse the string into a list
                if not isinstance(edge_list, list):
                    raise ValueError("The 'graph' field is not of type list after parsing")
            except Exception as e:
                print(f"Failed to parse the 'graph' field for sample {i}: {e}")
                continue

            max_edges = args.length_2
            # Keep only the last max_edges edges
            limited_edges = edge_list[-max_edges:]

            # Convert back to string format
            graph = str(limited_edges)

            print(graph)
            prompt += (
                f"Here is the {ordinal_words[i]} example: Question: Given a directed trust graph with edges {graph}. "
                f"What is the trust level from node {src} to node {dst} at time {timestamp}? Answer: {label}\n"
            )
        return prompt

    # Repeat initialization for non-flag=4
    ordinal_words = ["first", "second", "third", "fourth", "fifth", "sixth"]
    prompt = ""
    example_idx = 0
    num_0 = 0
    num_1 = 0

    successful_examples = 0
    counter = 0

    while successful_examples < 3:
        row = df.sample(1).iloc[0]
        src = int(row['src'])
        dst = int(row['dst'])
        label = int(row['label'])
        timestamp = row['timestamp']
        context_df = df[(df['timestamp'] <= timestamp) & (df.index < row.name)]

        shot_graph = build_graph_from_edges(context_df)

        # Determine subgraph construction case (tag)
        if src in shot_graph and dst in shot_graph:
            tag = 1
        elif src in shot_graph:
            tag = 2
        elif dst in shot_graph:
            tag = 3
        else:
            tag = 4

        if tag != flag:
            continue

        subgraph, tag = shot_subgraph_generation(context_df, src, dst, args.length_1, args.hop, args.length_2)

        if subgraph == -1:
            continue

        if label == 0:
            num_0 += 1
        else:
            num_1 += 1

        successful_examples += 1

        edge_list = [(int(u), int(v), int(data['label']), float(data['timestamp']))
                     for u, v, data in subgraph.edges(data=True)]
        prompt += (
            f"Here is the {ordinal_words[example_idx]} example: Question: Given a directed trust graph with edges [{edge_list}]. "
            f"What is the trust level from node {src} to node {dst} at time {timestamp}? Answer: {label}\n"
        )
        example_idx += 1

    print(f'example0: {num_0}, example1: {num_1}')
    return prompt


def update_result(results, case, label, is_correct):
    label_key = f"label{label}"

    # Initialize a specific case
    if case not in results:
        results[case] = {
            'label0': {'total': 0, 'correct': 0},
            'label1': {'total': 0, 'correct': 0}
        }

    results[case][label_key]['total'] += 1
    if is_correct:
        results[case][label_key]['correct'] += 1


from sklearn.metrics import f1_score, balanced_accuracy_score

def print_results(results):
    for case in sorted(results.keys()):
        # Collect true and predicted labels for each class
        true_labels = []
        predicted_labels = []

        print(f"Case {case}: ", end="")

        # Iterate over label0 and label1
        for label in ['label0', 'label1']:
            total = results[case][label]['total']
            correct = results[case][label]['correct']
            acc = correct / total if total > 0 else 0

            true_label = 0 if label == 'label0' else 1
            wrong_label = 1 - true_label

            # Construct true and predicted labels
            true_labels.extend([true_label] * total)
            predicted_labels.extend([true_label] * correct + [wrong_label] * (total - correct))

            print(f"{label} - total: {total}, correct: {correct}, acc: {acc:.2f}; ", end="")

        # Compute F1-macro and Balanced Accuracy
        if len(true_labels) > 0:
            f1_macro = f1_score(true_labels, predicted_labels, average='macro')
            ba_acc = balanced_accuracy_score(true_labels, predicted_labels)
            print(f"F1-macro: {f1_macro:.2f}, BA-Acc: {ba_acc:.2f}", end="")
        print()
