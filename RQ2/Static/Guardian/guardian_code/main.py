import numpy as np
from gcn import GCNTrainer
from arg_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, best_printer #save_logs

def main():
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    
    best = [["Run","Epoch","F1_micro","AUC","F1_macro","F1_weighted","MAE", "RMSE", "Training Time", "Inference Time"]]

    for t in range(10):
        trainer = GCNTrainer(args, edges)
        trainer.setup_dataset()
        print("Ready, Go! Round = " + str(t))
        trainer.create_and_train_model()

        if args.test_size > 0:
            best_epoch = [0, 0, 0, 0, 0, 0, 0, 0]
            for i in trainer.logs["performance"][1:]:
                if float(i[1]) > best_epoch[1]:
                    best_epoch = i

            best_epoch.append(trainer.logs["training_time"][-1][1])
            best_epoch.append(trainer.logs["inference_time"][-1][1])
            best_epoch.insert(0, t+1)
            best.append(best_epoch)

        args.seed += 1
    print("\nBest results of each run")
    best_printer(best)
    
    print("\nMean, Max, Min, Std")
    analyze = np.array(best)[1:,1:].astype(np.float64)
    mean = np.mean(analyze, axis=0)
    maxi = np.amax(analyze, axis=0)
    mini = np.amin(analyze, axis=0)
    std = np.std(analyze, axis=0)
    results = [["Epoch","F1_micro","AUC","F1_macro","F1_weighted","MAE", "RMSE", "Training Time", "Inference Time"], mean, maxi, mini, std]

    best_printer(results)

if __name__ == "__main__":
    main()