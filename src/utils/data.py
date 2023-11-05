import torch
import itertools

from torch.utils.data import Dataset


class ScDataset(Dataset):
    def __init__(self, exp, tpy, batch_size):
        super(ScDataset, self).__init__()

        if not isinstance(exp, torch.Tensor):
            raise Exception('exp must be a tensor')
        if not isinstance(tpy, torch.Tensor):
            raise Exception('tpy must be a tensor')
        if not isinstance(batch_size, int):
            raise Exception('smi must be a int')

        self.exp = exp
        self.tpy = tpy
        self.batch_size = batch_size
        self.idx = []

    def __getitem__(self, item):
        return self.exp[item, :], self.tpy[item, :]

    def __len__(self):
        return self.exp.size(0)

    def __re_idx(self):
        return itertools.product(self.idx, self.idx)
