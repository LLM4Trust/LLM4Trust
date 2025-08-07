import pandas as pd
import networkx as nx
from tqdm import tqdm

def check_paths_with_node_check(train_file, test_file, output_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    G = nx.DiGraph()
    G.add_edges_from(zip(train_df['src'], train_df['dst']))
    all_nodes = set(G.nodes)

    reachable_flags = []
    missing_node_flags = []

    print("Checking path reachability (including node existence)...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        src, dst = row['src'], row['dst']
        if src not in all_nodes or dst not in all_nodes:
            reachable_flags.append(False)
            missing_node_flags.append(True)
        else:
            reachable = nx.has_path(G, src, dst)
            reachable_flags.append(reachable)
            missing_node_flags.append(False)

    # Add two new columns
    test_df['reachable'] = reachable_flags
    test_df['missing_node'] = missing_node_flags

    # Summary statistics
    total = len(test_df)
    reachable_count = sum(test_df['reachable'])
    unreachable_count = total - reachable_count
    missing_node_count = sum(test_df['missing_node'])

    print(f"\n==== Path Statistics ====")
    print(f"Total number of test edges     : {total}")
    print(f"Reachable edges                : {reachable_count}")
    print(f"Unreachable edges              : {unreachable_count}")
    print(f"Edges with missing nodes       : {missing_node_count}")
    print(f"Reachability ratio             : {reachable_count / total:.2%}")

    test_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    check_paths_with_node_check(
        train_file="graph/train_edges.csv",
        test_file="graph/test_edges.csv",
        output_file="graph/test_edges_with_path_info.csv"
    )
