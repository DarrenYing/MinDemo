"""
PredRNN
无流水线并行
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import oneflow as flow
from oneflow.utils import data
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from models.PredRNN import PredRNN, PredRNNGraph
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"

P01 = flow.placement("cuda", [0, 1])
P0 = flow.placement("cuda", [0])
S0 = flow.sbp.split(0)
B = flow.sbp.broadcast

parser = get_parser()

args = parser.parse_args()

num_hidden = [64, 64, 64, 64]


def schedule_sampling():
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      ))
    return zeros


def train_graph():
    # init logger
    logger = log.get_logger(flow.env.get_rank(), [0])

    # load dataset
    train_dataset = FakeDataset()

    train_dataloader = flow.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger.print("dataset loaded")

    train_dataloader = tqdm(train_dataloader)
    totol_batch = len(train_dataloader)

    # init model and graph
    num_layers = 4
    model = PredRNN(num_layers, num_hidden, args.patch_size, args.total_length, args.input_length,
                    org_width=args.img_width).to(DEVICE)
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    model = model.to_global(placement=P01, sbp=B)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    base_graph = PredRNNGraph(model, sgd, args)
    base_graph.debug(1)
    logger.print("model loaded")

    # train
    total_loss = 0
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.from_numpy(reshape_patch(batch_data, args.patch_size))
            batch_data = flow.tensor(batch_data, dtype=flow.float32, placement=P01, sbp=S0)

            mask = flow.from_numpy(schedule_sampling())
            mask = flow.tensor(mask, dtype=flow.float32, placement=P01, sbp=S0)

            loss = base_graph(batch_data, mask)
            total_loss += loss.sum().item()
            if batch_idx % args.display_interval == 0:
                logger.print(f'epoch: {epoch}, batch: {batch_idx} / {totol_batch}, loss: {loss}')


def train_eager():
    # init logger
    logger = log.get_logger(flow.env.get_rank(), [0])

    # load dataset
    train_dataset = FakeDataset()

    train_dataloader = flow.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    logger.print("dataset loaded")

    train_dataloader = tqdm(train_dataloader)
    totol_batch = len(train_dataloader)

    # init model and graph
    num_layers = 4
    model = PredRNN(num_layers, num_hidden, args.patch_size, args.total_length, args.input_length,
                    org_width=args.img_width).to(DEVICE)
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    model = model.to_global(placement=P01, sbp=B)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = flow.nn.MSELoss().to(DEVICE)

    logger.print("model loaded")

    # train
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.from_numpy(reshape_patch(batch_data, args.patch_size))
            batch_data = flow.tensor(batch_data, dtype=flow.float32, placement=P01, sbp=S0)
            mask = flow.from_numpy(schedule_sampling())
            mask = flow.tensor(mask, dtype=flow.float32, placement=P01, sbp=S0)

            output = model(batch_data, mask)
            loss = loss_fn(output, batch_data[:, 1:])

            # Backpropagation
            sgd.zero_grad()
            loss.backward()
            sgd.step()

            logger.print(loss)


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
    elif args.run_mode == 'eager':
        train_eager()

