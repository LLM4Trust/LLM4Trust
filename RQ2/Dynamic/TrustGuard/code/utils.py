import torch
import numpy as np
from texttable import Texttable
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, accuracy_score


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def read_graph(args):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    edges = {}
    ecount = 0
    ncount = []
    edg = []
    lab = []
    with open(args.edge_path) as dataset:
        for edge in dataset:
            ecount += 1
            ncount.append(edge.split()[0])
            ncount.append(edge.split()[1])
            edg.append(list(map(float, edge.split()[0:2])))
            lab.append(list(map(float, edge.split()[2:])))
    edges["labels"] = np.array(lab)
    edges["edges"] = np.array(edg)
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount))
    #print("awdasd",edges)
    return edges


def setup_features(args):
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :return X: Node features.
    """
    # otc: (5881,64), alpha: (3783,64)
    # use random embeddings, can also generate embeddings using node2vec
    np.random.seed(args.seed)
    if args.data_path == "../data/bitcoinotc.csv":
        embedding = np.random.normal(0, 1, (5881, 64)).astype(np.float32)
    else:
        embedding = np.random.normal(0, 1, (3783, 64)).astype(np.float32)
    return embedding


def calculate_auc(scores, label):
    label_vector = [i for line in label for i in range(len(line)) if line[i] == 1]
    # print("wocaosocre",scores)

    prediction_vector = torch.argmax(scores, dim=1)

    acc_balanced = balanced_accuracy_score(label_vector, prediction_vector.cpu())
    mcc = matthews_corrcoef(label_vector, prediction_vector.cpu())
    # acc = accuracy_score(label_vector, prediction_vector.cpu())

    f1_micro = f1_score(label_vector, prediction_vector.cpu(), average="micro")
    f1_macro = f1_score(label_vector, prediction_vector.cpu(), average="macro")
    f1_weighted = f1_score(label_vector, prediction_vector.cpu(), average="weighted")
    # print("prediction_vector:", prediction_vector)

    prediction_pr = scores[:,1]       #scores: torch.Size([732, 2])
    # prediction_pr:torch.Size([732])
    # print("prediction_pr:", prediction_pr)
    # print("label_vector:", label_vector)
    auc = roc_auc_score(label_vector, prediction_pr.cpu().detach().numpy())   #, multi_class='ovr'
    # print("label_vector:", len(label_vector))    #label_vector: 732
    # precision = average_precision_score(label_vector, prediction_pr.cpu().detach().numpy())  # average precision

    return mcc, auc, acc_balanced, f1_weighted, f1_micro, f1_macro


def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable()
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 2 == 0])
    print(t.draw())

def best_printer(log):
    t = Texttable()
    t.set_precision(4)
    t.add_rows([per for per in log])
    print(t.draw())
