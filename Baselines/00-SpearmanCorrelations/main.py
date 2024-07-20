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
def spearman_corr(x):
    x_np = x.numpy()
    corr, _ = scipy.stats.spearmanr(x_np, axis=0)
    return torch.tensor(corr)

for data, id, group in tqdm(dataloader):
    data = data.squeeze(0)
    causality = spearman_corr(data).numpy()
    folder = "ECNs_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, causality)