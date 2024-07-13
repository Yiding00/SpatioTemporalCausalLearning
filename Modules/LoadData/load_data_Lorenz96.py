import pandas as pd
from .utils import MyDataset
from torch.utils.data import DataLoader
import numpy as np
def get_dataloader(batch_size, num_nodes, F, time_length, parent):
    '''
    torch.Size([batch_size, 10, num_nodes])
    '''
    data_list = []
    id_list = []
    group_list = []
    dir = "../"*parent+"data/Lorenz96/Lorenz96_node"+str(num_nodes)+"_F"+str(F)+"_T"+str(time_length)+".npy"

    X = np.load(dir)
    id_list = group_list = X

    dataset = MyDataset(X, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
