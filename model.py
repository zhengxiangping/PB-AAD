import torch
import torch.nn as nn
# from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv,DeepGraphInfomax,Linear,global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj,dense_to_sparse,to_dense_adj
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GINConv(in_channels, hidden_channels),
            GINConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.ReLU(hidden_channels),
            nn.ReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    def __init__(self,gnn_type,emb_dim,node_fea_dim,dropout_ratio,num_layer) -> None:
        super().__init__()
        self.dropout_ratio=dropout_ratio
        self.num_layer=num_layer
        self.batch_norm=nn.ModuleList()
        self.gnn=nn.ModuleList()
        self.feature_transform=nn.Linear(node_fea_dim,emb_dim)
        self.dropout=nn.Dropout(p=dropout_ratio)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        for i in range(num_layer):
            if gnn_type=='gin':
                self.gnn.append(GINConv((nn.Sequential(nn.Linear(emb_dim, emb_dim)))))
            elif gnn_type=='gcn':
                self.gnn.append(GCNConv(emb_dim,emb_dim))
            elif gnn_type=='gat':
                self.gnn.append(GATConv(emb_dim,emb_dim))
            elif gnn_type=='graphsage':
                self.gnn.append(SAGEConv(emb_dim,emb_dim))
            elif gnn_type=='dgi':
                DeepGraphInfomax(hidden_channels=emb_dim, encoder=Encoder(emb_dim, emb_dim),
                                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                                corruption=corruption)
            else:
                raise ValueError("Gnn type error!")
            self.batch_norm.append(nn.BatchNorm1d(emb_dim))
        # self.lin = nn.Linear(2*emb_dim,1)
        self.mlp = nn.Linear(emb_dim,1)
    
    def forward(self,x,edge_index):
        # x,edge_index= data.x,data.edge_index
        # edge_index, _ = dropout_adj(edge_index,p=0.9)
        # print(data.edge_index)
        x = self.relu(self.feature_transform(x))
        for layer in range(self.num_layer):
            x = self.gnn[layer](x,edge_index)
            x = self.batch_norm[layer](x)
            x = self.dropout(x)
            x = self.relu(x)
        # x = torch.cat([x[root_n_id], global_mean_pool(x, batch)], dim=-1)
        # x = self.dropout(self.relu(self.lin(x)))
        x = self.sigmoid(self.dropout(self.mlp(x)))
        return x
class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    >>> gcn.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        # if True:
        #     if utils.is_sparse_tensor(adj):
        #         adj = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         adj = utils.normalize_adj_tensor(adj)
        # else:
        #     adj_norm = adj
        
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return torch.sigmoid(x)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


class Classifier(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, final_dropout, neighbor_pooling_type, device):
        super(Classifier, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.linear_prediction = nn.Linear(hidden_dim, 1)
        self.final_dropout = final_dropout
        
    def forward(self, seq1, adj):
        h_1 = self.gin(seq1, adj)
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)
        return score_final_layer
class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
        
        #If sum or average pooling
        pooled = torch.spmm(Adj_block, h)
        if self.neighbor_pooling_type == "average":
            #If average pooling
            degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
            
            pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    
    def forward(self, feats, adj):
        h = feats
        for layer in range(self.num_layers):
            h = self.next_layer(h, layer, Adj_block = adj)

        return h
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)