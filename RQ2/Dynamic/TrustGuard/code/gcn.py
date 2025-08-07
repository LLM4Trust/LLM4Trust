import time
import torch
import numpy as np
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc, setup_features
from convolution import GraphConvolutionalNetwork, AttentionNetwork
from dataset import get_snapshot_index


class TrustGuard(torch.nn.Module):
    def __init__(self, device, args, X, num_labels, mask):
        super(TrustGuard, self).__init__()
        self.args = args
        torch.manual_seed(self.args.seed)  # fixed seed == 42
        self.device = device
        self.X = X
        self.dropout = self.args.dropout
        self.num_labels = num_labels
        self.mask = mask
        self.build_model()
        self.regression_weights = Parameter(torch.Tensor(self.args.layers[-1]*2, self.num_labels))
        init.xavier_normal_(self.regression_weights)  # initialize regression_weights

    def build_model(self):
        """
        Constructing spatial and temporal layers.
        """
        self.structural_layer = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        self.temporl_layer = AttentionNetwork(input_dim=self.args.layers[-1],n_heads=self.args.attention_head,num_time_slots=self.args.train_time_slots,attn_drop=0.5,residual=True)

    def calculate_loss_function(self, z, train_edges, target):
        """
        Calculating loss.
        :param z: Node embedding.
        :param train_edges: [2, #edges]
        :param target: Label vector storing 0 and 1.
        :return loss: Value of loss.
        """
        start_node, end_node = z[train_edges[0], :], z[train_edges[1], :]

        features = torch.cat((start_node, end_node), 1)

        # masking
        features = features[self.mask]
        target = target[self.mask]

        predictions = torch.mm(features, self.regression_weights)

        try:  # deal with imbalance data
            class_weight = torch.FloatTensor(1 / np.bincount(target.cpu()) * features.size(0))
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
            loss_term = criterion(predictions, target)
        except RuntimeError as e:
            if "weight tensor should be defined either for all" in str(e):
                print("Warning: Incomplete class weights detected. Using unweighted loss instead.")
                criterion = torch.nn.CrossEntropyLoss().to(self.device)
                loss_term = criterion(predictions, target)
            else:
                raise e

        return loss_term

    def forward(self, train_edges, y, y_train, index_list):
        structural_out = []
        index0 = 0
        for i in range(self.args.train_time_slots):
            structural_out.append(self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            index0 = index_list[i]

        structural_out = torch.stack(structural_out)
        structural_out = structural_out.permute(1,0,2)  # [N,T,F] [5881,7,32]
        temporal_all = self.temporl_layer(structural_out)  # [N,T,F]
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()  # [N,F]

        loss = self.calculate_loss_function(temporal_out, train_edges, y)

        return loss, temporal_out


class GCNTrainer(object):
    """
    Object to train and score the TrustGuard, log the model behaviour and save the output.
    """
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_start_time = time.time()
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["training_time"] = [["Epoch", "Seconds"]]
        self.logs["inference_time"] = [["Epoch", "Seconds"]]
        self.logs["case0"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["case1"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["case2"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        self.logs["case3"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]

    def setup_dataset(self):
        """
        Creating training snapshots and testing snapshots.
        """
        #self.index_list = get_snapshot_index(self.args.time_slots, data_path=self.args.data_path)
        total_samples = len(self.edges['edges'])
        self.index_list = get_snapshot_index(self.args.train_time_slots, data_path=self.args.train_path)
        print(self.index_list)

        train_index_t = self.index_list[-1]   #self.index_list[self.args.time_slots - 1]  train_index_t 19345
        print("train_index_t",train_index_t)
        self.train_edges = self.edges['edges'][:train_index_t]
        self.y_train = self.edges['labels'][:train_index_t]

        self.test_edges = self.edges['edges'][train_index_t:]
        self.y_test = self.edges['labels'][train_index_t:]
        print("self.test_edges", self.test_edges.shape)
        print("self.y_test ", self.y_test.shape)

        self.X = setup_features(self.args)  # Setting up the node features as a numpy array.
        self.num_labels = np.shape(self.y_train)[1]

        self.y = torch.from_numpy(self.y_train[:,1]).type(torch.long).to(self.device)
        # convert vector to number 0/1, 0 represents trust and 1 represents distrust

        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)  # (2, #edges)
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        self.X = torch.from_numpy(self.X).to(self.device)


    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        ## randomly sample edges from the training set
        # torch.manual_seed(self.args.seed)
        # num_samples = 400
        # perm = torch.randperm(self.y_train.size(0))[:num_samples]
        # mask = torch.zeros(self.y_train.size(0), dtype=torch.bool)
        # mask[perm] = True

        ## without masking
        mask = torch.ones(self.y_train.size(0), dtype=torch.bool)

        self.model = TrustGuard(self.device, self.args, self.X, self.num_labels, mask).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, desc="Loss")
        for epoch in epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, final_embedding = self.model(self.train_edges, self.y, self.y_train, self.index_list)
            loss.backward()
            epochs.set_description("TrustGuard (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch + 1, time.time() - start_time])
            self.score_model(epoch)

        self.logs["training_time"].append(["Total", time.time() - self.global_start_time])

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        start_time = time.time()
        self.model.eval()
        loss, self.train_z = self.model(self.train_edges, self.y, self.y_train, self.index_list)
        #loss, self.train_z = self.model(self.train_edges, self.sorted_train_edges, self.y, self.y_train)
        score_edges = torch.from_numpy(np.array(self.test_edges, dtype=np.int64).T).type(torch.long).to(self.device)  #self.obs self.test_edges
        # print("score_edges",score_edges)
        test_z = torch.cat((self.train_z[score_edges[0, :], :], self.train_z[score_edges[1, :], :]), 1)
        # score_edges[0, :] is the index of trustors, while score_edges[1, :] is the index of trustees
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))
        self.logs["inference_time"].append([epoch + 1, time.time() - start_time])
        predictions = F.softmax(scores, dim=1)  # (#test,2)

        # self.train_edges (2, #edges), self.test_edges (#edges, 2)
        flag = check_trustor_trustee_in_train(self.train_edges.cpu().numpy(), self.test_edges)

        # print evaluation result for each case
        print('\nmcc, auc, acc_balanced, precision, f1_micro, f1_macro')
        for case in range(4):
            idx = (flag == case)
            mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions[idx], self.y_test[idx])  # self.y_test_obs     self.y_test
            print('Case %d:' % case, end=' ')
            print('%.4f' % mcc, '%.4f' % auc, '%.4f' % acc_balanced, '%.4f' % precision, '%.4f' % f1_micro, '%.4f' % f1_macro)
            self.logs["case%d" % case].append([epoch + 1, mcc, auc, acc_balanced, precision, f1_micro, f1_macro])

        mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, self.y_test) #self.y_test_obs     self.y_test
        # print('\nmcc, auc, acc_balanced, precision, f1_micro, f1_macro')
        print('All:   ', '%.4f' % mcc, '%.4f' % auc, '%.4f' % acc_balanced, '%.4f' % precision, '%.4f' % f1_micro, '%.4f' % f1_macro)

        self.logs["performance"].append([epoch + 1, mcc, auc, acc_balanced, precision, f1_micro, f1_macro])


def check_trustor_trustee_in_train(train_tensor, test_tensor):
    # 0 Neither trustor nor trustee is in the training set
    # 1 Only the trustor is in the training set
    # 2 Only the trustee is in the training set
    # 3 Both trustor and trustee are in the training set
    # Extract trustors and trustees from the training set
    train_trustors = train_tensor[0]
    train_trustees = train_tensor[1]

    # Extract trustors and trustees from the test set
    test_trustors = test_tensor[:, 0]
    test_trustees = test_tensor[:, 1]

    # Check whether each test trustor appears in the training set
    trustor_in_train = np.isin(test_trustors, train_trustors)
    trustee_in_train = np.isin(test_trustees, train_trustees)

    # Generate flag based on the two boolean values
    flag = trustor_in_train.astype(int) * 1 + trustee_in_train.astype(int) * 2
    return flag

