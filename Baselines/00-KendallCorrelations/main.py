import torch
import numpy as np
import sys
sys.path.append("../..")
import time
from Modules.LoadData.load_data_ADNI import get_dataloader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.set_default_dtype(torch.float)
dataloader = get_dataloader(batch_size = 1, parent=3)
import os
import scipy.stats
folder_name = "ECNs_results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
def kendall_corr(x):
    x_np = x.numpy()
    num_vars = x_np.shape[1]
    corr_matrix = np.zeros((num_vars, num_vars))

    # 计算每一对变量之间的肯德尔秩相关系数
    for i in range(num_vars):
        for j in range(num_vars):
            if i <= j:
                tau, _ = scipy.stats.kendalltau(x_np[:, i], x_np[:, j])
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
            # 对角线上的值应该为1（一个变量与自身的相关性）
            if i == j:
                corr_matrix[i, j] = 1.0

    return torch.tensor(corr_matrix)

for data, id, group in tqdm(dataloader):
    data = data.squeeze(0)
    causality = kendall_corr(data)
    folder = "ECNs_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, causality)