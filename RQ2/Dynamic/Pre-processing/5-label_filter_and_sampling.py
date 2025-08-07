import pandas as pd

def filter_label_0_edges(input_file, output_file):
    # Load the test set
    df = pd.read_csv(input_file)

    # Filter edges with label == 0
    label_0_df = df[df['label'] == 0]

    # Print statistics
    total_label_0 = len(label_0_df)
    reachable_label_0 = label_0_df[label_0_df['reachable'] == True]
    unreachable_label_0 = label_0_df[label_0_df['reachable'] == False]
    missing_node_label_0 = label_0_df[label_0_df['missing_node'] == True]

    print(f"Total number of edges with label=0: {total_label_0}")
    print(f"Reachable edges           : {len(reachable_label_0)}")
    print(f"Unreachable edges         : {len(unreachable_label_0)}")
    print(f"Edges with missing nodes  : {len(missing_node_label_0)}")

    # Save edges with label = 0
    label_0_df.to_csv(output_file, index=False)
    print(f"Edges with label=0 saved to: {output_file}")


def filter_label_1_edges(input_file, output_file):
    # Load the test set
    df = pd.read_csv(input_file)

    # Filter edges with label == 1
    label_1_df = df[df['label'] == 1]

    # Print statistics
    total_label_1 = len(label_1_df)
    reachable_label_1 = label_1_df[label_1_df['reachable'] == True]
    unreachable_label_1 = label_1_df[label_1_df['reachable'] == False]
    missing_node_label_1 = label_1_df[label_1_df['missing_node'] == True]

    print(f"Total number of edges with label=1: {total_label_1}")
    print(f"Reachable edges           : {len(reachable_label_1)}")
    print(f"Unreachable edges         : {len(unreachable_label_1)}")
    print(f"Edges with missing nodes  : {len(missing_node_label_1)}")

    # Save edges with label = 1
    label_1_df.to_csv(output_file, index=False)
    print(f"Edges with label=1 saved to: {output_file}")


if __name__ == "__main__":
    input_file = "graph/test_spec.csv"
    filter_label_1_edges(input_file, "graph/test_edges_label_1.csv")
    filter_label_0_edges(input_file, "graph/test_edges_label_0.csv")
