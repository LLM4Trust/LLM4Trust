import numpy as np
from gcn import GCNTrainer
from arg_parser import parameter_parser
from utils import tab_printer, read_graph, best_printer, setup_features,score_printer
from contextlib import redirect_stdout


def main():
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)  # number of edges --> otc: 35592, alpha: 24186
    setup_features(args)

    best = [["Run", "Epoch", "MCC", "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"]]
    case0_best = [["Run", "Epoch", "MCC", "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"]]
    case1_best = [["Run", "Epoch", "MCC", "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"]]
    case2_best = [["Run", "Epoch", "MCC", "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"]]
    case3_best = [["Run", "Epoch", "MCC", "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"]]

    runs = 5
    for t in range(runs):
        trainer = GCNTrainer(args, edges)
        trainer.setup_dataset()
        print("Ready, Go! Round = " + str(t))
        trainer.create_and_train_model()
        # score_printer(trainer.logs)
        best_epoch = [0, 0, 0, 0, 0, 0, 0, 0]
        case0 = [0, 0, 0, 0, 0, 0, 0, 0]
        case1 = [0, 0, 0, 0, 0, 0, 0, 0]
        case2 = [0, 0, 0, 0, 0, 0, 0, 0]
        case3 = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(1, len(trainer.logs["performance"][1:])):
            # sum of MCC, AUC, ACC_Balanced, F1_Macro
            if float(trainer.logs["performance"][i][3]+trainer.logs["performance"][i][6]) > (best_epoch[3]+best_epoch[6]):
                best_epoch = trainer.logs["performance"][i]
                case0 = trainer.logs["case0"][i]
                case1 = trainer.logs["case1"][i]
                case2 = trainer.logs["case2"][i]
                case3 = trainer.logs["case3"][i]

        best_epoch.append(trainer.logs["training_time"][-1][1])
        best_epoch.append(trainer.logs["inference_time"][-1][1])
        best_epoch.insert(0, t + 1)
        best.append(best_epoch)

        case0.append(trainer.logs["training_time"][-1][1])
        case0.append(trainer.logs["inference_time"][-1][1])
        case0.insert(0, t + 1)
        case0_best.append(case0)
        case1.append(trainer.logs["training_time"][-1][1])
        case1.append(trainer.logs["inference_time"][-1][1])
        case1.insert(0, t + 1)
        case1_best.append(case1)
        case2.append(trainer.logs["training_time"][-1][1])
        case2.append(trainer.logs["inference_time"][-1][1])
        case2.insert(0, t + 1)
        case2_best.append(case2)
        case3.append(trainer.logs["training_time"][-1][1])
        case3.append(trainer.logs["inference_time"][-1][1])
        case3.insert(0, t + 1)
        case3_best.append(case3)

        args.seed += 1

    ## print all results
    print_all(best, "All")
    print_all(case0_best, "Case 0 (both out)")  # this is the case 4 defined in the paper
    print_all(case1_best, "Case 1 (trustor)")  # this is the case 2 defined in the paper
    print_all(case2_best, "Case 2 (trustee)")  # this is the case 3 defined in the paper
    print_all(case3_best, "Case 3 (both in)")  # this is the case 1 defined in the paper


def _print_all(best, label):
    print("\n{}\tBest results of each run".format(label))
    best_printer(best)
    print("Mean, Max, Min, Std")
    analyze = np.array(best)[1:, 1:].astype(np.float64)
    mean = np.mean(analyze, axis=0)
    maxi = np.amax(analyze, axis=0)
    mini = np.amin(analyze, axis=0)
    std = np.std(analyze, axis=0)
    results = [["Epoch", 'MCC', "AUC", "ACC_Balanced", "F1_Weighted", "F1_Micro", "F1_Macro", "Training Time", "Inference Time"], mean, maxi, mini, std]

    best_printer(results)


def print_all(best, label, file_path="./result_otc.txt"):
    with open(file_path, 'a') as f:
        with redirect_stdout(f):
            _print_all(best, label)


if __name__ == "__main__":
    main()
