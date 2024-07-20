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
folder_name = "ECNs_results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
def pearson_corr(x):
    # 确保输入是浮点型
    x = x.float()
    # 计算均值
    mean_x = x.mean(dim=0, keepdim=True)
    # 计算去均值后的数据
    xm = x - mean_x
    # 计算标准差
    std_x = x.std(dim=0, unbiased=False, keepdim=True)
    std_x[std_x == 0] = 1.0
    # 计算归一化后的数据
    xm = xm / std_x
    # 计算相关矩阵
    r = torch.matmul(xm.T, xm) / (x.size(0))
    return r

for data, id, group in tqdm(dataloader):
    data = data.squeeze(0)
    causality = pearson_corr(data).numpy()
    folder = "ECNs_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, causality)