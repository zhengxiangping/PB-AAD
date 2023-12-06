'''
Author: wubo2180 15827403235@163.com
Date: 2022-07-06 17:22:00
LastEditors: wubo2180 15827403235@163.com
LastEditTime: 2022-07-11 17:51:42
FilePath: \python workspace\test1\baselines.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from turtle import forward
import torch.nn.functional as F
from pygod.models import DOMINANT,GAAN,CONAD,ANEMONE,OCGNN,AnomalyDAE
# class DOMINANT_(DOMINANT):
#     def __init__(self,
#                  hid_dim=64,
#                  num_layers=4,
#                  dropout=0.3,
#                  weight_decay=0.,
#                  act=F.relu,
#                  alpha=None,
#                  contamination=0.1,
#                  lr=5e-3,
#                  epoch=5,
#                  gpu=0,
#                  batch_size=0,
#                  num_neigh=-1,
#                  verbose=False):
#         super(DOMINANT_,self).__init__(self,
#                  hid_dim=64,
#                  num_layers=4,
#                  dropout=0.3,
#                  weight_decay=0.,
#                  act=F.relu,
#                  alpha=None,
#                  contamination=0.1,
#                  lr=5e-3,
#                  epoch=5,
#                  gpu=0,
#                  batch_size=0,
#                  num_neigh=-1,
#                  verbose=False)
# import sys,torch
# from pygod.utils import load_data
# from utils import load_data as load_data_

# # sys.exit()
# # print(data_)
# data = load_data('inj_cora','./')
# data.y = data.y.bool()
# # data,split_idx=load_data_('./','wiki',10)
# # data.train_mask=torch.arange(100).long()
# print(data)


model = DOMINANT (hid_dim=128,epoch=1)
# print(model)
# _,embedding=model.fit(data)
# print(embedding)
# # sys.exit()
# labels = model.predict(data)
# print('Labels:')
# print(labels,labels.shape)
# outlier_scores = model.decision_function(data)
# print('Raw scores:')
# print(outlier_scores)
# prob = model.predict_proba(data)
# print('Probability:')
# print(prob)
# labels, confidence = model.predict(data, return_confidence=True)
# print('Labels:')
# print(labels)
# print('Confidence:')
# print(confidence)

# from pygod.metrics import eval_roc_auc
# # print(data.label[:,0])
# # auc_score = eval_roc_auc(data.y.numpy(), outlier_scores[data.label[:,0]])
# auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
# print('AUC Score:', auc_score)

# embedding = embedding.detach().cpu().numpy()
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import numpy as np
# X=np.arange(data.x.shape[0])
# # np.random.shuffle(X)
# print(embedding.shape)
# y=X
# clf = SVC(C=2)
# X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.33, random_state=42)
# print(X_train, X_test)
# print(X_train.shape)
# clf.fit(embedding[X_train], data.y.numpy()[X_train]) 
# score = clf.decision_function(embedding[X_train])
# score = clf.decision_function(embedding[y_train])
# # print('score',score)

# print('roc_auc',roc_auc_score(data.y.numpy()[y_train],score))


import sys,torch
from pygod.utils import load_data
from utils import load_data as load_data_
from torch_geometric.nn import GINConv,GCNConv
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
# sys.exit()
# print(data_)
data = load_data('inj_cora','./')
# data.y = data.y.bool()
data.y = data.y.float()
data.y[data.y>0] = 1.0
# data,split_idx=load_data_('./','reddit',10)
# data.train_mask=torch.arange(100).long()
print(data.x.dtype,data.y.dtype,data.edge_index.dtype)
device= torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GCNConv(data.x.shape[1],128)
        self.layer2 = GCNConv(128,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sig = nn.Sigmoid()
    def forward(self,x,edge_index):
        x = self.layer1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x, edge_index)
        return self.sig(x)
model=GCN().to(device)
print(model)
epochs=100
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion=nn.BCELoss()
data=data.to(device)
model.train()
for i in range(epochs):
    optimizer.zero_grad()
    x = model(data.x,data.edge_index)
    # print(x.shape)
    # print(data.y.shape)
    loss = criterion(x[:1000].squeeze(),data.y[:1000])
    loss.backward()
    optimizer.step()
model.eval()
x = model(data.x,data.edge_index)
x = x[1000:].squeeze().detach().cpu().numpy()
y = data.y[1000:].detach().cpu().numpy()
print((x>1).sum(),(y>1.0).sum())
auc = roc_auc_score(y,x)
print(auc)
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.targeted_attack import Nettack
sys.exit()
model = DOMINANT (hid_dim=128,epoch=100)
print(model)
_,embedding=model.fit(data)
# print(embedding)
# sys.exit()
labels = model.predict(data)
print('Labels:')
print(labels,labels.shape)
outlier_scores = model.decision_function(data)
print('Raw scores:')
print(outlier_scores)
prob = model.predict_proba(data)
print('Probability:')
print(prob)
labels, confidence = model.predict(data, return_confidence=True)
print('Labels:')
print(labels)
print('Confidence:')
print(confidence)

from pygod.metrics import eval_roc_auc
# # print(data.label[:,0])
# auc_score = eval_roc_auc(data.y.numpy(), outlier_scores[data.label[:,0]])
auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
print('AUC Score:', auc_score)

# embedding = embedding.detach().cpu().numpy()
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import numpy as np
# X=np.arange(data.x.shape[0])
# np.random.shuffle(X)
# print(embedding.shape)

# clf = SVC(C=2)
# X_train, X_test, y_train, y_test = train_test_split(X, X, test_size=0.33, random_state=42)
# # print(X_train, X_test)
# # print(X_train.shape)
# mean=[]
# for id, (train_idx, test_idx) in enumerate(split_idx):
#     clf.fit(embedding[data.label[train_idx,0]], data.y.numpy()[train_idx]) 
#     # score = clf.decision_function(embedding[train_idx])
#     score = clf.decision_function(embedding)
#     # print('score',score)
#     roc_auc = roc_auc_score(data.y.numpy()[test_idx],score[data.label[test_idx,0]])
#     mean.append(roc_auc)
#     print('roc_auc',roc_auc)
# print('mean roc_auc',np.mean(mean))