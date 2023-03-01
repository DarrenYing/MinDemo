"""
测试混合并行

python3 -m oneflow.distributed.launch --nproc_per_node 2 demo01.py

在数据并行中64是每个进程每次读取的数据量大小，
使用两个gpu进行数据并行的时候，会将两个进程各64的数据拼到一起作为global_batch_size，
训练时每轮用128的数据更新模型，所以 50000/(64+64) 是使用该数据集时每个epoch的迭代次数，
50000/64 是所有进程每个epoch读取数据的总次数

"""
import flowvision
import flowvision.transforms as transforms
from oneflow.utils import data
import oneflow as flow
import oneflow.nn as nn
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

BATCH_SIZE = 64
EPOCH_NUM = 100

PLACEMENT = flow.placement("cuda", [0, 1])
S0 = flow.sbp.split(0)
B = flow.sbp.broadcast

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="/workspace/projects/data",
    train=True,
    transform=transforms.ToTensor(),
    download=False,
)

sampler = data.DistributedSampler(training_data, shuffle=False)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, sampler=sampler
)

print("dataset loaded")

model = flowvision.models.mobilenet_v2().to(DEVICE)
model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
model = model.to_global(placement=PLACEMENT, sbp=B)

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)


for t in range(EPOCH_NUM):
    sampler.set_epoch(t)
    total = correct = 0
    print(f"Epoch {t + 1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to_global(placement=PLACEMENT, sbp=S0)
        y = y.to_global(placement=PLACEMENT, sbp=S0)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += y.size(0)
        correct += flow.eq(pred.argmax(dim=1), y).sum().item()

        current = batch * BATCH_SIZE
        if batch % 50 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] | acc: {100.0 * correct / total:.3f}")
