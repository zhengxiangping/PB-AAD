import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset, download_url
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch_geometric.utils import k_hop_subgraph,to_scipy_sparse_matrix,from_scipy_sparse_matrix,to_dense_adj,dropout_adj
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
from deeprobust.graph import utils
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))
def truncatedSVD( data, k=50):
    """Truncated SVD on input data.

    Parameters
    ----------
    data :
        input matrix to be decomposed
    k : int
        number of singular values and vectors to compute.

    Returns
    -------
    numpy.array
        reconstructed matrix.
    """
    print('=== GCN-SVD: rank={} ==='.format(k))
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        print("rank_after = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print("rank_before = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
        print("rank_after = {}".format(len(diag_S.nonzero()[0])))

    return U @ diag_S @ V
@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt

@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt
def drop_dissimilar_edges(features, adj, metric='similarity'):
    """Drop dissimilar edges.(Faster version using numba)
    """
    binary_feature=False
    threshold=0.01
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    else:
        if binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold=threshold)
    print('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj
def load_data(root,datasets,num_folds):
    # load the adjacency
    adj = np.loadtxt(root+'/data/'+datasets+'.txt')
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print('Load the edge_index done!')
    # print('adj',adj)
    # load the user label
    label = np.loadtxt(root+'/data/'+datasets+'_label.txt')
    y = label[:, 1]
    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)
    print('Number of nodes: ', nb_nodes)
    
    # split the train_set and validation_set

    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        # print(test_idx)
        # print(test_idx.shape)
        split_idx.append((train_idx, test_idx))
    # load initial features
    feats = np.load(root+'/features/'+datasets+'_feature64.npy')
    # print('feats',feats,feats.shape)
    edge_index=torch.from_numpy(edge_index).long()
    # print(edge_index.shape)
    # print(to_scipy_sparse_matrix(edge_index).shape[0])
    # print(truncatedSVD(to_scipy_sparse_matrix(edge_index)))
    # edge_index = truncatedSVD(to_scipy_sparse_matrix(edge_index),k=20)
    # print(edge_index)
    # edge_index = drop_dissimilar_edges(feats,to_scipy_sparse_matrix(edge_index))
    # # edge_index = to_scipy_sparse_matrix(edge_index)
    # if sp.issparse(edge_index):
    #     edge_index = sparse_mx_to_torch_sparse_tensor(edge_index)
    # else:
    #     edge_index = torch.FloatTensor(edge_index)
    # if True:
    #     if utils.is_sparse_tensor(edge_index):
    #         edge_index = utils.normalize_adj_tensor(edge_index, sparse=True)
    #     else:
    #         edge_index = utils.normalize_adj_tensor(edge_index)
    # else:
    #     adj_norm = adj
    # # edge_index = to_dense_adj(edge_index)
    edge_index,_ = dropout_adj(edge_index,p=0.2)
    print(edge_index)
    # dd
    # edge_index = torch.FloatTensor(edge_index)
    # print(to_dense_adj(edge_index)[0].shape)
    # dd
    # edge_index = to_dense_adj(edge_index)[0]
    # print(edge_index)
    # edge_index, _ = from_scipy_sparse_matrix(edge_index)
    # print(edge_index.shape)
    # dd
    feats=torch.from_numpy(feats).float()
    label=torch.from_numpy(label).float()
    data=Data(x=feats,edge_index=edge_index,y=label[:,1].reshape(-1,1),nb_nodes=nb_nodes,label=label.long())
    # print(max(edge_index[0]))
    # print(max(edge_index[1]))
    # k_hop_subgraph([0,1],2,data.edge_index)
    return data,split_idx,nb_nodes

def plot_xy(x_values, label, title,name):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    # sns.scatterplot(x="x", y="y", hue="label", data=df,sizes=(40, 40))
    plt.scatter(x_values[:,0],x_values[:,1], c=label)
    plt.legend(['Abnormal','Normal'],fontsize=15)
    # plt.title(title)
    plt.xlabel("",fontsize=40)
    plt.ylabel("",fontsize=40)
    
    plt.savefig(name+'.pdf')
    plt.show()


def plot_tsne(x_value, y_value,name):
    # x_value, y_value = get_data()
    # PCA 降维
    
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x_value)
    # plot_xy(x_pca, y_value, "PCA")
    # t-sne 降维
    
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_value)
    plot_xy(x_tsne, y_value, "t-sne",name)
    
def cluster(x_value, y_value,name):

    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_value)
    print(x_tsne)
    # ddd
    y_pred = KMeans(n_clusters=50, random_state=0).fit_predict(x_tsne)
    print(y_pred)
    # ddd
    plt.scatter(x_tsne[:,0],x_tsne[:,1], c=y_pred)
    plt.savefig(name+'.pdf')
    plt.show()
    # ddd