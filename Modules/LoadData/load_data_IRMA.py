import pandas as pd
from .utils import MyDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size, parent):
    '''
    torch.Size([batch_size, 9, 5])
    '''
    data_list = []
    id_list = []
    group_list = []
    if parent==3:
        dir = "../../../data/IRMA/"
    elif parent==2:
        dir = "../../data/IRMA/"  
    elif parent==1:
        dir = "../data/IRMA/"
    else:
        dir = None
        print("Invalid parent")
    exp_num = {
        "on": 8,
        "off": 10,
    }
    for i in ["on","off"]:
        for j in range(1,exp_num[i]+1):
            folder = dir+"IRMA_all/Switch_"+i+"_"+str(j)+".txt"
            row_data = pd.read_csv(folder, sep='\s+').values
            data_list.append(row_data)
            id_list.append(i+"_"+str(j))
            group_list.append(i)

    dataset = MyDataset(data_list, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
