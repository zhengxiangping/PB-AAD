'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-14 22:26:42
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-07-20 23:07:42
FilePath: \python_workspace\GPT-GNN-ATEST\basic\data\dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import torch
from torch.utils.data import Dataset

from ..utils import load_pkl
import random
import numpy as np
import math
class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, device,transform=None) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        self.device = device
        self.transform = transform
        # self.target_transform = target_transform

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """
        
        idx = list(self.index[index])

        # print(idx)
        # dd
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]

        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
        if self.transform:
            history_data,history_data_mask = self.transform(history_data)
            return future_data, history_data, history_data_mask
        else:
            return future_data, history_data
        # if self.target_transform:
        #     future_data = self.target_transform(future_data)
        # return future_data, history_data, history_data_mask
        
    def __len__(self):
        """Dataset length
        Returns:
            int: dataset length
        """
        return len(self.index)


