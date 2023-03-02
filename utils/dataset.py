import oneflow as flow
from oneflow.utils import data


class FakeDataset(data.Dataset):

    def __getitem__(self, index):
        return flow.randn(4, 1, 200, 200)

    def __len__(self):
        return 1000
