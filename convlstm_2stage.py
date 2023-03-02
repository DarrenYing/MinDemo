"""
ConvLSTM dp2+pp2 测试

python3 -m oneflow.distributed.launch --nproc_per_node 4 convlstm_2stage.py
"""


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import oneflow as flow
from oneflow.utils import data
from tqdm import tqdm
import numpy as np

from models.TwoStageConvLSTM import ConvLSTMPipeline, ConvLSTMGraph, Stage0Model, Stage1Model
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser

BROADCAST = [flow.sbp.broadcast]
S0 = flow.sbp.split(dim=0)
P01 = flow.placement("cuda", ranks=[0, 1])
P23 = flow.placement("cuda", ranks=[2, 3])
P0123 = flow.placement("cuda", ranks=[0, 1, 2, 3])


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
                      ))

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
    num_layers = 2
    s0_model = Stage0Model(num_layers, num_hidden[:2], args)
    s1_model = Stage1Model(num_layers, num_hidden[2:], args)
    model = ConvLSTMPipeline(s0_model, s1_model, args, [P01, P23])
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    graph_pipeline = ConvLSTMGraph(model, sgd, args, [P01, P23])
    graph_pipeline.debug(1)

    # train
    # loss_fn = flow.nn.MSELoss()
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.from_numpy(reshape_patch(batch_data, args.patch_size))
            batch_data = batch_data.to_global(placement=P0123, sbp=S0).to_global(placement=P01, sbp=S0)

            _, mask = schedule_sampling(1.0, epoch)
            mask = flow.from_numpy(mask)
            mask = mask.to_global(placement=P0123, sbp=S0).to_global(placement=P01, sbp=S0)

            loss = graph_pipeline(batch_data, mask)
            print(loss)

            # output = model(batch_data, mask)
            # print(f"output: {output.shape}")
            # loss = loss_fn(output, batch_data[:, 1:])
            # print(f"loss: {loss}")


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
