import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from typing import Callable, Tuple


class GIN_layer(MessagePassing):
    """图同构网络层实现，适用于批处理张量操作
    
    该层实现了GIN (Graph Isomorphism Network)的核心操作：
    x′ = h_θ((1 + ε)x + Σ x_j)，其中j是节点i的邻居
    
    Args:
        nn (torch.nn.Module): 映射节点特征的神经网络，例如MLP
        eps (float): epsilon值，默认为0
        train_eps (bool): 是否将epsilon设为可训练参数，默认False
    
    Input:
        - 输入为(x, Adj)元组：
          * x: 形状为[B, T, N, D]的节点特征张量
            B=批量大小，T=时间步，N=节点数，D=特征维度
          * Adj: 形状为[N, N]的邻接矩阵
    
    Output:
        - 输出与输入相同形式的(x_new, Adj)元组
    """
    def __init__(self, nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        
        # 设置epsilon为可训练或固定参数
        if train_eps:
            self.eps = nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """前向传播
        
        Args:
            input: (x, Adj)元组，x是节点特征，Adj是邻接矩阵
            
        Returns:
            更新后的(x, Adj)元组
        """
        x, Adj = input
        
        # 计算GIN聚合：(1+eps)*x + 邻居聚合
        x = (1 + self.eps) * x + torch.einsum('hi,btij->bthj', Adj, x)
        
        # 应用变换网络
        x = self.nn(x)
        
        return (x, Adj)

    def __repr__(self) -> str:
        """字符串表示"""
        return f'{self.__class__.__name__}(nn={self.nn})'