from argparse import ArgumentParser
import random
import numpy as np

parser = ArgumentParser()
parser.add_argument("--run", type=int, default=1)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--model", type=str, default='claude-3-7-sonnet-20250219')
parser.add_argument("--knowledge", type=int, default=1)
parser.add_argument("--example", type=int, default=2, help="0 for zero-shot; 1 for one-shot; 2 for few-shot")
parser.add_argument("--cot", type=int, default=0)
parser.add_argument("--role", type=int, default=1)
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--hop', type=int, default=3)
parser.add_argument('--length_1', type=int, default=50)
parser.add_argument('--length_2', type=int, default=50)
parser.add_argument("--train_file", type=str, default='graph/train_edges.csv')
parser.add_argument("--output_file", type=str, default='output-200-42.txt')

args = parser.parse_args()
# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)

from prompt import DyGraphPrompt
from api import send_prompt
from utlis import save_file
from utlis import subgraph_generation
from utlis import build_graph_from_edges
from utlis import print_results
import time
import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from utlis import update_result

start_time = time.time()
flag = 0

for run in range(args.run):  # Repeat for specified number of runs
    print("-----------------run {}-----------------".format(run))

    test_file = "graph/test_spec.csv"  # Includes case 1/2/3
    test_df = pd.read_csv(test_file)
    test_df = test_df.sample(n=args.epoch, random_state=args.seed).reset_index(drop=True)

    count_label_0 = 0
    count_label_1 = 0
    count_error = 0
    an_label_0 = 0
    an_label_1 = 0
    y_true = []
    y_pred = []

    results = {}
    time_list = []

    for i, row in test_df.iterrows():
        src = int(row['src'])
        dst = int(row['dst'])
        label = int(row['label'])
        timestamp = row['timestamp']

        train_file = args.train_file
        output_file = args.output_file
        max_1 = args.length_1
        hop = args.hop
        max_2 = args.length_2

        tag = 0

        train_df = pd.read_csv(train_file)
        G_train = build_graph_from_edges(train_df)

        if src in G_train and dst in G_train:
            tag = 1
        elif src in G_train:
            tag = 2
        elif dst in G_train:
            tag = 3
        else:
            tag = 4

        G = subgraph_generation(train_file, src, dst, max_1, hop, max_2)

        while G == -1:
            new_row = test_df.sample(n=1).iloc[0]
            src = int(new_row['src'])
            dst = int(new_row['dst'])
            label = int(new_row['label'])
            timestamp = new_row['timestamp']
            # Retry subgraph generation
            G = subgraph_generation(train_file, src, dst, max_1, hop, max_2)

        edge_list = [(int(u), int(v), int(data['label']), float(data['timestamp'])) for u, v, data in G.edges(data=True)]
        prompt = DyGraphPrompt(args)
        prompt_qa = prompt.generate_qa(edge_list, src, dst, label, timestamp, tag)
        save_file(output_file, prompt_qa)  # Save the prompt
        output_answer, time_once = send_prompt(args, prompt_qa)
        time_list.append(time_once)
        output = int(output_answer)

        if label == 0:
            count_label_0 += 1
        else:
            count_label_1 += 1
        y_true.append(label)

        y_pred.append(output)
        standard_answer = prompt_qa["answer"]

        update_result(results, tag, label, output_answer == standard_answer)

        # Check output correctness
        if output_answer == standard_answer:
            flag += 1
            if label == 0:
                an_label_0 += 1
            else:
                an_label_1 += 1

        print("epoch:", i + 1, "output:", output_answer, "truth:", standard_answer, output_answer == standard_answer)

print("{}/{}".format(flag, args.run * args.epoch), flag/(args.run * args.epoch))

print("Total time:", sum(time_list))
print("label=0:","{}/{}".format(an_label_0,  count_label_0))
print("label=1:","{}/{}".format(an_label_1, count_label_1))
print("error:","{}".format(count_error))

print("y_true", y_true)
print("y_pred", y_pred)

acc_balanced = balanced_accuracy_score(y_true, y_pred)
print("Balanced Accuracy:", acc_balanced)

f1 = f1_score(y_true, y_pred, average='macro')
print("F1 Score:", f1)

mcc = matthews_corrcoef(y_true, y_pred)
print("MCC:", mcc)

print_results(results)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
