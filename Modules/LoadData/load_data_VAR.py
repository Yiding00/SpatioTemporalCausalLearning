import pandas as pd
from .utils import MyDataset
from torch.utils.data import DataLoader
import numpy as np
def get_dataloader(batch_size, num_nodes, lag, parent):
    '''
    torch.Size([batch_size, 10, num_nodes])
    '''
    data_list = []
    id_list = []
    group_list = []
    if parent==3:
        dir = "../../../data/VAR/"
    elif parent==2:
        dir = "../../data/VAR/"  
    elif parent==1:
        dir = "../data/VAR/"
    else:
        dir = None
        print("Invalid parent")

    dir = "../../../data/VAR/VAR_node"+str(num_nodes)+"_lag"+str(lag)+".npy"
    X = np.load(dir)
    id_list = group_list = X

    dataset = MyDataset(X, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
