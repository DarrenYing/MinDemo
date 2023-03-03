"""
ConvLSTM dp4 测试

python3 -m oneflow.distributed.launch --nproc_per_node 4 convlstm_dp.py
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import oneflow as flow
from oneflow.utils import data
from tqdm import tqdm
import numpy as np

from models.BaseConvLSTM import BaseConvLSTM, BaseConvLSTMGraph
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser

BROADCAST = [flow.sbp.broadcast]
P0123 = flow.placement("cuda", ranks=[0, 1, 2, 3])
DEVICE = "cuda" if flow.cuda.is_available() else "cpu"


parser = get_parser()

args = parser.parse_args()

gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

num_hidden = [64, 64, 64, 64]


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      ), dtype=np.float32)

    return 0, zeros


def train_graph():
    # init logger
    logger = log.get_logger(flow.env.get_rank(), [0])

    # load dataset
    train_dataset = FakeDataset()

    sampler = data.DistributedSampler(train_dataset, drop_last=True)

    train_dataloader = flow.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True
    )

    logger.print("dataset loaded")

    if flow.env.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader)

    totol_batch = len(train_dataloader)

    # init model and graph
    num_layers = 4
    model = BaseConvLSTM(num_layers, num_hidden, args).to(DEVICE)
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    model = model.to_global(placement=P0123, sbp=BROADCAST)

    sgd = flow.optim.Adam(model.parameters(), lr=0.001)
    # sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    graph_pipeline = BaseConvLSTMGraph(model, sgd, args)
    graph_pipeline.debug(1)

    # train
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.from_numpy(reshape_patch(batch_data, args.patch_size))
            batch_data = batch_data.to_global(placement=P0123, sbp=BROADCAST)

            _, mask = schedule_sampling(1.0, epoch)
            mask = flow.from_numpy(mask)
            mask = mask.to_global(placement=P0123, sbp=BROADCAST)

            loss = graph_pipeline(batch_data, mask)
            print(loss)


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
