from torch.utils.data import Dataset, DataLoader
import torch

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())
        self.values = list(data.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.values[idx]
        tensor_value = torch.tensor(value, dtype=torch.float32)
        return key, tensor_value

class MyDataset_ADNI(Dataset):
    def __init__(self, data, id, group):
        self.data = data
        self.id = id
        self.group = group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id = self.id[idx]
        data_value = self.data[idx]
        data = torch.tensor(data_value, dtype=torch.float32)
        group = self.group[idx]
        return data, id, group

