import sys
import pandas as pd
from .utils import MyDataset_ADNI
from torch.utils.data import DataLoader
import pandas as pd 
import numpy as np

def get_dataloader(batch_size):
    data_list = []
    id_list = []
    group_list = []
    df = pd.read_csv("../../../data/ADNI-adhd/label.csv")

    # id, group, sex, age, visit
    # group: CN, EMCI, LMCI, MCI, SMC
    # sex: F, M
    for row in df.itertuples():
        folder = "../../../data/ADNI-adhd/" + row.group + "/" + row.id + ".txt"
        row_data = pd.read_csv(folder, sep='\s+', header=None).values
        data_list.append(row_data)
        id_list.append(row.id)
        group_list.append(row.group)
    dataset = MyDataset_ADNI(data_list, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
