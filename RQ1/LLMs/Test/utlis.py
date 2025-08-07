import json
import pickle
import networkx as nx
import os


def save_file(args, content):
    filename = "output_{}_{}.txt".format(args.model, args.task)
    # filename = "output_{}_{}_temperature0.txt".format(args.model, args.task)

    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, indent=4))
        f.write("\n")


def load_graph_from_json(args, filename):
    folder = "save_graphs_{}".format(args.task)
    filepath = os.path.join(folder, filename)

    if args.task == "dynamic":
        with open(filepath, "r") as f:
            graph_data = json.load(f)

        edges = graph_data["edges"]
        special_nodes = graph_data["special_nodes"]
        time = graph_data["time"]

        return edges, special_nodes, time

    elif args.task == "asymmetric":
        with open(filepath, "r") as f:
            graph_data = json.load(f)

        edges = graph_data["edges"]
        special_nodes = graph_data["special_nodes"]
        weight = graph_data["answer"]

        return edges, special_nodes, weight

    elif args.task == "propagative":
        with open(filepath, "r") as f:
            graph_data = json.load(f)

        edges = graph_data["edges"]
        special_nodes = graph_data["special_nodes"]
        min_weight = graph_data["min_weight"]

        return edges, special_nodes, min_weight

    elif args.task == "composable":
        with open(filepath, "r") as f:
            graph_data = json.load(f)

        edges = graph_data["edges"]
        special_nodes = graph_data["special_nodes"]
        num_min = graph_data["num_min"]
        num_max = graph_data["num_max"]

        return edges, special_nodes, num_min, num_max

    elif args.task == "context":
        with open(filepath, "r") as f:
            graph_data = json.load(f)

        edges = graph_data["edges"]
        special_nodes = graph_data["special_nodes"]
        c = graph_data["c"]
        w = graph_data["w"]
        return edges, special_nodes, c, w
