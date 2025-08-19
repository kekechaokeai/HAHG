import numpy as np
import torch
from utils import load_data, set_params, evaluate, run_kmeans
from module.HAHG import *
from module.graph_generating import *
from module.preprocess import *
import warnings
import datetime
import time
import pickle as pkl
import random
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')
args = set_params()
args.load_parameters = False
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train():
    feat, adjs, label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print('n_clusters',nb_classes)
    print(args)

    if torch.cuda.is_available():
        print('Using CUDA')
        adjs = [adj.cuda() for adj in adjs]
        feat = feat.cuda()

    N = feat.size(0)
    adj_I = torch.eye(N).to(device)
    adjs_o = [normalize_adj_from_tensor(adj_I + adj.to_dense(), mode='sym').to_sparse() for adj in adjs]
    model = HAHG(feats_dim, sub_num, args.hidden_dim, args.embed_dim, args.tau, args.dropout, args.act, args.drop_feat, len(feat), args.dataset, args.alpha, args.nlayer,args.graph_k,args.k_pos,args.beta, args.fusion, args)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)
    if torch.cuda.is_available():
        model.cuda()
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    period = 100
    best_epoch=0
    auc_list = {'0':[], '1':[], '2':[]}
    ma_list = {'0':[], '1':[], '2':[]}
    mi_list = {'0':[], '1':[], '2':[]}
    nmi_list = []
    ari_list = []

    starttime = datetime.datetime.now()


    for epoch in range(args.nb_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(adjs, feat)
        loss = loss
        loss.backward()
        optimizer.step()

        print("Epoch:", epoch)
        print('Total loss: ', loss.item())
        # print("Attention:", att)

        if best > loss.item():
            best = loss.item()
            cnt_wait = 0
            best_epoch = epoch
            torch.save(model.state_dict(), './checkpoint/'+args.dataset+'/best_'+str(args.seed)+'.pth')
        else:
            cnt_wait += 1
        if cnt_wait >= args.patience:
            break


    model.cuda()
    epoch = best_epoch
    print("---------------------------------------------------")
    model.eval()
    embeds = model.get_embeds(feat, adjs_o)

    nmi, ari = run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset,
                          epoch + 1)
    nmi_list.append(nmi)
    ari_list.append(ari)

    for i in range(len(idx_train)):

        ma,mi,auc = evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device,
                       args.dataset,
                       args.eva_lr, args.eva_wd, starttime, epoch + 1)
        auc_list[str(i)].append(auc)
        ma_list[str(i)].append(ma)
        mi_list[str(i)].append(mi)

    for i in range(len(args.ratio)):
        index = auc_list[str(i)].index(max(auc_list[str(i)]))
        f = open("result_" + args.dataset + str(args.ratio[i]) + ".txt", "a")
        #f = open("no_semantic_" + args.dataset + str(args.ratio[i]) + ".txt", "a")
        f.write(str(starttime.strftime('%Y-%m-%d %H:%M')) + "\t")
        f.write("Ma-F1_MAX: " + str(np.around(ma_list[str(i)][index], 2)) + "\t")
        f.write(" Mi-F1_MAX: " + str(np.around(mi_list[str(i)][index], 2)) +  "\t")
        f.write(" AUC_MAX: " + str(np.around(auc_list[str(i)][index], 2)) +  "\n")
        f.write(str(args) +  "\n")
        f.close()

        f = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
        #f = open("no_senmantic_" + args.dataset + "_NMI&ARI.txt", "a")
        f.write(str(starttime.strftime('%Y-%m-%d %H:%M')) + "\t Corresponding auc ratio "+str(args.ratio[i])+" \t MAX NMI: " + str(np.round(nmi_list[index], 4)) + \
                "\t MAX ARI: " + str(np.round(ari_list[index], 4)) + "\n")
        f.close()


    f = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
    #f = open("no_semantic_" + args.dataset + "_NMI&ARI.txt", "a")    
    f.write(str(starttime.strftime('%Y-%m-%d %H:%M')) +"\t MAX NMI: " + str(np.round(max(nmi_list),4)) +\
         "\t MAX ARI: " + str(np.round(max(ari_list),4)) + "\n")
    f.write(str(args) + "\n")
    f.close()


if __name__ == '__main__':

    set_seed(args.seed)
    train()

# if __name__ == '__main__':
#     for drop_rate in [0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8,0.9]:
#         args.node_drop = drop_rate
#         print(f"\n===== Running with node_drop = {drop_rate} =====")
#         set_seed(args.seed)
#         train()
