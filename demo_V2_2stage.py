"""
PredRNN形式的ConvLSTM流水线并行测试
实现了两阶段流水线实际训练
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import oneflow as flow
from oneflow import nn
from tqdm import tqdm
import numpy as np

from models.TwoStageConvLSTM_V2 import PredRNNPipeline, PredRNNGraph, Stage0Model, Stage1Model
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser
from utils.loss_utils import LossRecoder

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", ranks=[0])
P1 = flow.placement("cuda", ranks=[1])

parser = get_parser()

args = parser.parse_args()

num_hidden = [64, 64, 64, 64]


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      ))
    return 0.0, zeros


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

    if flow.env.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader)

    totol_batch = len(train_dataloader)

    # init model and graph
    num_layers = 2
    s0_model = Stage0Model(num_layers, num_hidden[:2], args.patch_size)
    s1_model = Stage1Model(num_layers, num_hidden[2:], args.patch_size)
    model = PredRNNPipeline(s0_model, s1_model, args.total_length, args.input_length, num_hidden[-1], args.patch_size, [P0, P1])
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    graph_pipeline = PredRNNGraph(model, sgd, args, [P0, P1])
    # graph_pipeline.debug(1)

    # train
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.tensor(reshape_patch(batch_data, args.patch_size),
                                     dtype=flow.float32, placement=P0,
                                     sbp=BROADCAST)
            _, mask = schedule_sampling(1.0, epoch)
            mask = flow.tensor(mask, dtype=flow.float32, placement=P0, sbp=BROADCAST)
            loss = graph_pipeline(batch_data, mask)
            print(graph_pipeline)
            # logger.print(loss)
            # print(loss)
            # loss_aver = loss.sum().item() / args.batch_size

        flow.save(model.state_dict(), args.checkpoint_path, global_dst_rank=0)



def evaluate_model():
    # init model and graph
    num_layers = 2
    s0_model = Stage0Model(num_layers, num_hidden[:2], args.patch_size)
    s1_model = Stage1Model(num_layers, num_hidden[2:], args.patch_size)
    model = PredRNNPipeline(s0_model, s1_model, args.total_length, args.input_length,
                            num_hidden[-1], args.patch_size, [P0, P1])
    params = flow.load(args.checkpoint_path, global_src_rank=0)
    model.load_state_dict(params)

    test_set = FakeDataset()

    test_dataloader = flow.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    if flow.env.get_rank() == 0:
        test_dataloader = tqdm(test_dataloader)

    stats = {
        'batch_num': len(test_dataloader),
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
    }
    loss_recoder = LossRecoder(flow.env.get_rank(), [0], **stats)

    mse_criterion = flow.nn.MSELoss()
    mae_criterion = flow.nn.L1Loss()
    with flow.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            batch_data = flow.tensor(reshape_patch(batch_data, args.patch_size),
                                     dtype=flow.float32, placement=P0,
                                     sbp=BROADCAST)

            _, mask = schedule_sampling(1.0, 0)
            mask = flow.tensor(mask, dtype=flow.float32, placement=P0, sbp=BROADCAST)
            output = model(batch_data, mask)
            # output = output.detach().cpu().to_local()
            # batch_data = batch_data.detach().cpu().to_local()
            # loss_recoder.calc_scores(output, batch_data[:, 1:])
            mse_score = mse_criterion(output, batch_data[:, 1:])
            mae_score = mae_criterion(output, batch_data[:, 1:])
            if flow.env.get_rank() == 0:
                print(mse_score, mae_score)

    # loss_recoder.record()
    # loss_recoder.save(args.stats_path)


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
    elif args.run_mode == 'test':
        evaluate_model()
