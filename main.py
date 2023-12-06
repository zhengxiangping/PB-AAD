import argparse
import random
# from sklearn import metrics

from sklearn.metrics import roc_auc_score,recall_score
from sklearn.cluster import KMeans
from torch_geometric.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
import time
from progressbar import *
from utils import *
from model import *
import os,sys
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj,to_dense_adj,dense_to_sparse,is_undirected
from torch_geometric.loader import ShaDowKHopSampler
# from torch_geometric.nn import DataLoader
# from torch_geometric.data import DataLoader

    
def evaluate(model, data,test_idx):
    model.eval()
    pred=  model(data.x,data.edge_index)
    pred  =  pred[data.label[test_idx,0]].detach().cpu().numpy()
    target  =  data.y[test_idx].detach().cpu().numpy()
    auc  =  roc_auc_score(target,pred)
    
    return auc


def train(args,model,data,train_idx,test_idx,criterion,optimizer):
    test_auc = []
    loss_list = []
    
    for epoch in tqdm(range(1,args.epochs + 1),ncols=50):
        
        model.train()
        optimizer.zero_grad()
        output= model(data.x,data.edge_index)
        loss = criterion(output[data.label[train_idx,0]],data.y[train_idx])
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        test_auc.append(evaluate(model,data,test_idx))

    return np.max(test_auc),loss_list
def main(args):
    data,split_idx,nb_nodes = load_data(args.dir,args.dataset,args.num_folds)
    data = data.to(args.device)
    test_auc_list = []
    for id, (train_idx, test_idx) in enumerate(split_idx):
        criterion = nn.BCELoss()
        train_idx = torch.from_numpy(train_idx).long().to(args.device)
        test_idx = torch.from_numpy(test_idx).long().to(args.device)
        net = Discriminator(args.gnn_type,args.emb_dim,args.node_fea_dim,args.dropout_ratio,args.num_layer).to(args.device)
        # net = GCN(args.node_fea_dim, args.emb_dim,1,device='cuda').to(args.device)
        # net = Classifier(args.num_layers, args.num_mlp_layers, args.node_fea_dim, args.emb_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
        print(net)
        optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = args.decay)
        
        test_auc, loss_list = train(args,net,data,train_idx,test_idx,criterion,optimizer)
        # test_auc = evaluate(net,data,test_idx)
        print('test_auc: {:.4f} '.format(test_auc))
        test_auc_list.append(test_auc)
        # test_recall_list.append(test_recall)
        # break
    print('mean test_auc: {:.4f} '.format(np.mean(test_auc_list)))
    # print('loss',loss_list[2])
    # label=data.label.cpu().numpy()
    # index=np.random.randint(0,label.shape[0]-1,size=(1000,))
    # print(embedding[label[:,0]][index],label[:,1][index])
    # plot_tsne(embedding[label[:,0]][index],label[:,1][index],name='trained')

if __name__  ==  "__main__":
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(
        description = 'PyTorch implementation')
    parser.add_argument('--device', type = int, default = 1,
                        help = 'which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type = str, default = 'amazon',
                        help = 'dataset name (wiki, reddit, alpha,amazon)')
    parser.add_argument('--dir', type = str, default = '.',
                        help = 'dataset directory')
    parser.add_argument('--delta', type = float, default = 0.1,
                        help = 'dataset directory')    
    parser.add_argument('--epochs', type = int, default = 300,
                        help = 'number of epochs to train (default: 100)')
    parser.add_argument('--decay', type = float, default = 0,
                        help = 'weight decay (default: 0)')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5,
                        help = 'dropout ratio (default: 0)')
    parser.add_argument('--emb_dim', type = int, default = 128,
                        help = 'embedding dimensions (default: 128)')
    parser.add_argument('--gnn_type', type = str, default = "gcn",
                        help = 'gin type (gin,gcn,graphsage,gat)')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = 'learning rate (default: 0.01)')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers for dataset loading')
    parser.add_argument('--num_layer', type = int, default = 2,
                        help = 'number of GNN message passing layers (default: 2).')
    parser.add_argument('--node_fea_dim', type = int, default = 64,
                        help = 'node feature dimensions (BIO: 2; DBLP: 10; CHEM: ))')
    parser.add_argument('--num_folds', type = int, default = 10,
                        help = 'number of folds (default: 10)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over neighboring nodes: sum or average')
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print(args)
    main(args)
