import pandas as pd

def filter_reachable_label_0_edges(input_file, output_file):
    # Load test edges with label=0
    df = pd.read_csv(input_file)

    # Filter edges with label=0 and reachable
    reachable_label_0_df = df[(df['label'] == 0) & (df['reachable'] == True)]

    # Print statistics
    total_reachable_label_0 = len(reachable_label_0_df)
    print(f"Total number of edges with label=0 and reachable: {total_reachable_label_0}")

    # Save reachable label=0 edges
    reachable_label_0_df.to_csv(output_file, index=False)
    print(f"Reachable label=0 edges saved to: {output_file}")

def filter_unreachable_label_0_edges(input_file, output_file):
    # Load test edges with label=0
    df = pd.read_csv(input_file)

    # Filter edges with label=0 and unreachable
    unreachable_label_0_df = df[(df['label'] == 0) & (df['reachable'] == False)]

    # Print statistics
    total_unreachable_label_0 = len(unreachable_label_0_df)
    print(f"Total number of edges with label=0 and unreachable: {total_unreachable_label_0}")

    # Save unreachable label=0 edges
    unreachable_label_0_df.to_csv(output_file, index=False)
    print(f"Unreachable label=0 edges saved to: {output_file}")

def filter_unreachable_label_1_edges(input_file, output_file):
    # Load test edges with label=1
    df = pd.read_csv(input_file)

    # Filter edges with label=1 and unreachable
    unreachable_label_1_df = df[(df['label'] == 1) & (df['reachable'] == False)]

    # Print statistics
    total_unreachable_label_1 = len(unreachable_label_1_df)
    print(f"Total number of edges with label=1 and unreachable: {total_unreachable_label_1}")

    # Save unreachable label=1 edges
    unreachable_label_1_df.to_csv(output_file, index=False)
    print(f"Unreachable label=1 edges saved to: {output_file}")

def filter_reachable_label_1_edges(input_file, output_file):
    # Load test edges with label=1
    df = pd.read_csv(input_file)

    # Filter edges with label=1 and reachable
    reachable_label_1_df = df[(df['label'] == 1) & (df['reachable'] == True)]

    # Print statistics
    total_reachable_label_1 = len(reachable_label_1_df)
    print(f"Total number of edges with label=1 and reachable: {total_reachable_label_1}")

    # Save reachable label=1 edges
    reachable_label_1_df.to_csv(output_file, index=False)
    print(f"Reachable label=1 edges saved to: {output_file}")

if __name__ == "__main__":
    input_file = "graph/test_spec.csv"

    filter_reachable_label_1_edges(input_file, "graph/test_edges_label_1_reachable.csv")
    filter_unreachable_label_1_edges(input_file, "graph/test_edges_label_1_unreachable.csv")
    filter_reachable_label_0_edges(input_file, "graph/test_edges_label_0_reachable.csv")
    filter_unreachable_label_0_edges(input_file, "graph/test_edges_label_0_unreachable.csv")
