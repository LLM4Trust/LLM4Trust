import pandas as pd
import os

def deduplicate_edges(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    print(f"Total number of original edges: {len(df)}")

    # Check for duplicate edges (only consider 'src' and 'dst', ignoring 'label' and 'timestamp')
    duplicate_mask = df.duplicated(subset=['src', 'dst'], keep=False)
    num_duplicates = duplicate_mask.sum()

    if num_duplicates > 0:
        print(f"Number of duplicate edges found: {num_duplicates}")
    else:
        print("No duplicate edges found.")

    # For each (src, dst) pair, keep the one with the latest timestamp
    dedup_df = df.sort_values('timestamp').groupby(['src', 'dst'], as_index=False).last()

    print(f"Number of edges after deduplication: {len(dedup_df)}")

    # Sort by timestamp in ascending order
    dedup_df = dedup_df.sort_values('timestamp', ascending=True)

    # Save to a new file
    dedup_df.to_csv(output_file, index=False)
    print(f"Deduplicated and sorted edges saved to: {output_file}")

if __name__ == "__main__":
    # Replace with your actual file path
    os.makedirs("graph", exist_ok=True)

    # Set file paths
    input_path = "bitcoinotc.csv"
    output_path = "graph/1-edges_dedup_sorted.csv"

    deduplicate_edges(input_path, output_path)
