'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-14 22:27:47
LastEditors: wubo2180 15827403235@163.com
LastEditTime: 2024-07-31 17:11:31
FilePath: \python_workspace\GPT-GNN-ATEST\basic\losses\losses.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn.functional as F

from ..utils import check_nan_inf
import numpy as np

def l1_loss(input_data, target_data, **kwargs):
    """unmasked mae."""

    return F.l1_loss(input_data, target_data)


def l2_loss(input_data, target_data, **kwargs):
    """unmasked mse"""

    check_nan_inf(input_data)
    check_nan_inf(target_data)
    return F.mse_loss(input_data, target_data)

def sce_loss(preds, labels, alpha=3, null_val: float = np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    preds = F.normalize(preds, p=2, dim=-1)
    labels = F.normalize(labels, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)
    ###
    loss = (1 - (preds * labels).sum(dim=-1)).pow_(alpha)
    # loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # loss = loss.mean()
    return torch.mean(loss)