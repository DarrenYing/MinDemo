"""
MotionRNN+PredRNN
无流水线并行
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import oneflow as flow
from oneflow.utils import data
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from models.MotionRNN_PredRNN import MotionRNN, MotionRNNGraph
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser
from utils.loss_utils import LossRecoder

device = flow.device("cuda:0")

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
    # init summary writer
    run_dir = args.tb_summary_path
    tb = SummaryWriter(run_dir)

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
    model = MotionRNN(num_layers, num_hidden, args).to(device)
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    base_graph = MotionRNNGraph(model, sgd, args)
    base_graph.debug(1)

    logger.print("model loaded")

    mse_criterion = flow.nn.MSELoss()
    # train
    total_loss = 0
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.tensor(reshape_patch(batch_data, args.patch_size),
                                     dtype=flow.float32).to(device)
            _, mask = schedule_sampling(1.0, epoch)
            mask = flow.tensor(mask, dtype=flow.float32).to(device)
            # eager模式
            # output = model(batch_data, mask)
            # sgd.zero_grad()
            # loss = mse_criterion(output, batch_data[:, 1:])
            # loss.backward()
            # sgd.step()
            # logger.print(loss)

            # graph模式
            loss = base_graph(batch_data, mask)
            loss_aver = loss.sum().item() / args.batch_size
            total_loss += loss.sum().item()
            if batch_idx % args.display_interval == 0:
                logger.print(f'epoch: {epoch}, batch: {batch_idx} / {totol_batch}, loss: {total_loss}')
                # logger.print(f'time: {step_time:.3f}, TFLOPS: {get_tflops_func(step_time):.3f}')
                tb.add_scalar('TrainLoss', loss_aver, batch_idx)
                tb.add_scalar('TrainLoss2', total_loss / batch_idx, batch_idx)

        flow.save(model.state_dict(), args.checkpoint_path)


def evaluate_model():
    # init model and graph
    num_layers = 4
    model = MotionRNN(num_layers, num_hidden, args).to(device)
    params = flow.load(args.checkpoint_path)
    model.load_state_dict(params)

    test_set = FakeDataset()

    test_dataloader = flow.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    test_dataloader = tqdm(test_dataloader)

    stats = {
        'batch_num': len(test_dataloader),
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
    }

    loss_recoder = LossRecoder(flow.env.get_rank(), [0], **stats)

    with flow.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            batch_data = flow.tensor(reshape_patch(batch_data, args.patch_size),
                                     dtype=flow.float32).to(device)
            _, mask = schedule_sampling(1.0, 0)
            mask = flow.tensor(mask, dtype=flow.float32).to(device)
            output = model(batch_data, mask)
            output = output.detach().cpu()
            batch_data = batch_data.detach().cpu()
            # output_image(output, args.patch_size, args.output_path)
            loss_recoder.calc_scores(output, batch_data[:, 1:])

        loss_recoder.record()
        loss_recoder.save(args.stats_path)


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
    elif args.run_mode == 'test':
        evaluate_model()

