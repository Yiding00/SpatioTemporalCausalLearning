import sys
import pandas as pd
from .utils import MyDataset_ADNI
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np

def get_dataloader(batch_size, parent):
    data_list = []
    id_list = []
    group_list = []
    if parent==3:
        dir = "../../../data/ADNI-adhd/"
    elif parent==2:
        dir = "../../data/ADNI-adhd/"  
    elif parent==1:
        dir = "../data/ADNI-adhd/"
    else:
        dir = None
        print("Invalid parent")
    df = pd.read_csv(dir +"label.csv")
    # id, group, sex, age, visit
    # group: CN, EMCI, LMCI, MCI, SMC
    # sex: F, M
    for row in df.itertuples():
        folder = dir + row.group + "/" + row.id + ".txt"
        row_data = pd.read_csv(folder, sep='\s+', header=None).values
        data_list.append(row_data)
        id_list.append(row.id)
        group_list.append(row.group)
    dataset = MyDataset_ADNI(data_list, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
