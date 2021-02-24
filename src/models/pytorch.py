from torch.utils.data import Dataset
import numpy as np
import torch

class PyTorchDataset(Dataset):
        
    def __init__(self, feat, targ):
        self.feat = self.to_tensor(feat.copy()) #astype(np.float32)
        self.targ = self.to_tensor(targ.copy()) #astype(np.float32)
        
    def __len__(self):
        return len(self.targ)
        
    def __getitem__(self, index):
        return self.feat[index], self.targ[index]
        
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))