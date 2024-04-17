import dgl
import torch
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):

    def __init__(self, g_list, *tensors, transform=None):
        assert all(len(g_list) == t.shape[0] for t in tensors), "Size mismatch between inputs"
        self.g_list = g_list
        self.tensors = tensors
        self.len = len(g_list)
        self.transform = transform

    def __getitem__(self, index):
        ret = (self.g_list[index],) + tuple(t[index] for t in self.tensors)
        if self.transform:
            ret = self.transform(ret)
        return ret

    def __len__(self):
        return self.len


class GraphDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        out = tuple(map(list, zip(*batch)))
        gs = dgl.batch(out[0])
        tensors = tuple([torch.stack(o) for o in out[1:]])
        return (gs,) + tensors