import torch
from torch.utils import data
# import Dataset, DataLoader

class InpData(data.Dataset):
    """"""

    def __init__(self, X, Y):
        # self.X = torch.from_numpy(X).float()
        # self.Y = torch.from_numpy(Y).float()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def random_classification(s, d):
    inputs = torch.randn(s, d)
    labels = torch.randint(2, (s,))

    datasets = {'train': InpData(inputs, labels),
                'val': InpData(inputs, labels)}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=20, shuffle=False, num_workers=0)
                   for x in ['train', 'val']}
    return dataloaders