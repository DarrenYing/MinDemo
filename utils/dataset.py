import oneflow as flow
from oneflow.utils import data


class FakeDataset(data.Dataset):

    def __getitem__(self, index):
        return flow.randn(20, 1, 400, 400)

    def __len__(self):
        return 100
