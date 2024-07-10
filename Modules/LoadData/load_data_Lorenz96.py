import pandas as pd
from .utils import MyDataset
from torch.utils.data import DataLoader
import numpy as np
def get_dataloader(batch_size, num_nodes, F, parent):
    '''
    torch.Size([batch_size, 10, num_nodes])
    '''
    data_list = []
    id_list = []
    group_list = []
    if parent==3:
        dir = "../../../data/Lorenz96/"
    elif parent==2:
        dir = "../../data/Lorenz96/"  
    elif parent==1:
        dir = "../data/Lorenz96/"
    else:
        dir = None
        print("Invalid parent")

    dir = "../../../data/Lorenz96/Lorenz96_node"+str(num_nodes)+"_lag"+str(F)+".npy"
    X = np.load(dir)
    id_list = group_list = X

    dataset = MyDataset(X, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
