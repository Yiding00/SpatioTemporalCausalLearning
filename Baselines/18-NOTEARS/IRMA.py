import torch
import numpy as np
import sys
sys.path.append("../..")
import time
from Modules.LoadData.load_data_IRMA import get_dataloader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.set_default_dtype(torch.float)
dataloader = get_dataloader(batch_size = 1, parent=3)

from utils import notears_nonlinear
from Modules.StaticDAG.NotearsMLP import NotearsMLP
# The first dim is nodes num
model = NotearsMLP(dims=[5, 10, 1], bias=True)
print("model = NotearsMLP(dims=[5, 10, 1], bias=True)")

start_time = time.time()
for data, id, group in tqdm(dataloader):
    causality = notears_nonlinear(model, data.numpy(), lambda1=0.01, lambda2=0.01)
    folder = "IRMA_results/" + id[0] + "_" + group[0] + ".npy"
    np.save(folder, causality)
end_time = time.time()
print(f"time:{end_time - start_time}seconds")