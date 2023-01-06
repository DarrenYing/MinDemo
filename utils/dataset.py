import oneflow as flow
from oneflow.utils import data


class FakeDataset(data.Dataset):

    def __getitem__(self, index):
        return flow.randn(20, 1, 900, 900)

    def __len__(self):
        return 100
