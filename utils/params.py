import argparse
import torch
import sys


dataset = 'acm'


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=2000)  # 400
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--meta_hidden_channels', type=int, default=128)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.2)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--graph_k', type=int, default=5)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--node_alpha', type=float, default=0.1)
    parser.add_argument('--node_drop', type=float, default=0.3)
    parser.add_argument('--node_out_channels', type=int, default=64)
    parser.add_argument('--n_clusters', type=int, default=3)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args

def cora_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=2000)  # 400
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--meta_hidden_channels', type=int, default=512)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.3)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--graph_k', type=int, default=5)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--node_alpha', type=float, default=0.1)
    parser.add_argument('--node_drop', type=float, default=0.3)
    parser.add_argument('--node_out_channels', type=int, default=64)

    args, _ = parser.parse_known_args()
    args.type_num = [298, 418, 818, 426, 217, 180, 351]  # the number of every node type
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=97)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--meta_hidden_channels', type=int, default=512)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_feat', type=float, default=0.1)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--graph_k', type=int, default=9)
    parser.add_argument('--k_pos', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--node_alpha', type=float, default=2)
    parser.add_argument('--node_drop', type=float, default=0.5)
 
    parser.add_argument('--node_out_channels', type=int, default=64)

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args



def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--meta_hidden_channels', type=int, default=128)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.003)
    parser.add_argument('--eva_wd', type=float, default=1e-4)

    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.6)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--graph_k', type=int, default=5)
    parser.add_argument('--k_pos', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--node_alpha', type=float, default=1.5)
    parser.add_argument('--node_out_channels', type=int, default=64)
    parser.add_argument('--node_drop', type=float, default=0.9)

    args, _ = parser.parse_known_args()
    args.type_num = [4278, 2081, 5257]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args




def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=2000)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=60)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer', type=int, default=2)
    parser.add_argument('--meta_hidden_channels', type=int, default=128)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.005)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--drop_feat', type=float, default=0.1)

    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--graph_k', type=int, default=10)
    parser.add_argument('--k_pos', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--node_alpha', type=float, default=2)
    parser.add_argument('--node_drop', type=float, default=0.6)
    parser.add_argument('--node_out_channels', type=int, default=64)

    args, _ = parser.parse_known_args()
    args.type_num = [2614, 1286, 4, 9]  # the number of every node type
    args.nei_num = 4  # the number of neighbors' types
    return args






def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "cora":
        args = cora_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == 'yelp':
        args = yelp_params()
    elif dataset == 'imdb':
        args = imdb_params()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return args
