import pandas as pd

def split_train_test(input_file, train_file, test_file, train_ratio=0.8):
    # Read the deduplicated and timestamp-sorted edge file
    df = pd.read_csv(input_file)
    total = len(df)
    split_index = int(total * train_ratio)

    # Split the dataset
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Save the splits
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Total number of edges: {total}")
    print(f"Training set size: {len(train_df)} saved to: {train_file}")
    print(f"Test set size: {len(test_df)} saved to: {test_file}")

if __name__ == "__main__":
    input_path = "graph/1-edges_dedup_sorted.csv"
    train_output = "graph/train_edges.csv"
    test_output = "graph/test_edges.csv"

    split_train_test(input_path, train_output, test_output)
