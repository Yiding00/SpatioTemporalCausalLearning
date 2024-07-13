import pandas as pd
from .utils import MyDataset
from torch.utils.data import DataLoader
import pandas as pd 

def get_dataloader(batch_size, parent):
    '''
    torch.Size([batch_size, 187, 90])
    '''
    data_list = []
    id_list = []
    group_list = []
    dir = "../"*parent+"data/ADNI/"

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
    dataset = MyDataset(data_list, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
