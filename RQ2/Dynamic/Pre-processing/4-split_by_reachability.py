import pandas as pd

def deduplicate_edges(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"Original total number of edges: {len(df)}")

    # Check for duplicate edges (only considering src and dst, ignoring label and timestamp)
    duplicate_mask = df.duplicated(subset=['src', 'dst'], keep=False)
    num_duplicates = duplicate_mask.sum()

    if num_duplicates > 0:
        print(f"Number of duplicate edges found: {num_duplicates}")
    else:
        print("No duplicate edges found.")

    # Keep the edge with the maximum timestamp for each (src, dst) pair
    dedup_df = df.sort_values('timestamp').groupby(['src', 'dst'], as_index=False).last()

    print(f"Number of edges after deduplication: {len(dedup_df)}")

    # Sort edges by timestamp in ascending order
    dedup_df = dedup_df.sort_values('timestamp', ascending=True)
    dedup_df.to_csv(input_file, index=False)
    print(f"Deduplicated and sorted edges have been overwritten to: {input_file}")


def split_test_edges_by_path_info(input_file):
    df = pd.read_csv(input_file)

    reachable_df = df[(df['reachable'] == True) & (df['missing_node'] == False)]
    unreachable_df = df[(df['reachable'] == False) & (df['missing_node'] == False)]
    missing_node_df = df[df['missing_node'] == True]

    reachable_df.to_csv("graph/reachable_edges.csv", index=False)
    unreachable_df.to_csv("graph/unreachable_edges.csv", index=False)
    missing_node_df.to_csv("graph/missing_node_edges.csv", index=False)

    print("Saved:")
    print(f"- Reachable edges        : {len(reachable_df)} -> reachable_edges.csv")
    print(f"- Unreachable edges      : {len(unreachable_df)} -> unreachable_edges.csv")
    print(f"- Edges with missing nodes: {len(missing_node_df)} -> missing_node_edges.csv")

    test_spec_df = pd.concat([reachable_df, missing_node_df], ignore_index=True)
    test_spec_df.to_csv("graph/test_spec.csv", index=False)
    print(f"- Merged test_spec edges : {len(test_spec_df)} -> test_spec.csv")

    deduplicate_edges("graph/test_spec.csv")


if __name__ == "__main__":
    split_test_edges_by_path_info("graph/test_edges_with_path_info.csv")
