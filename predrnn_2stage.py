"""
PredRNN流水线并行测试
实现了两阶段流水线实际训练
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

from models.TwoStagePredRNN import PredRNNPipeline, PredRNNGraph, Stage0Model, Stage1Model
from utils.dataset import FakeDataset
import utils.logger as log
from utils.utils import reshape_patch, get_parser
from utils.loss_utils import LossRecoder

BROADCAST = [flow.sbp.broadcast]
S0 = flow.sbp.split(dim=0)
P0 = flow.placement("cuda", ranks=[0, 1])
P1 = flow.placement("cuda", ranks=[2, 3])


parser = get_parser()

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  #args.gpus
gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

num_hidden = [64, 64, 64, 64]


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      ))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_graph():
    # init summary writer
    run_dir = args.tb_summary_path
    tb = SummaryWriter(run_dir)

    # init logger
    logger = log.get_logger(flow.env.get_rank(), [0])

    # load dataset
    train_dataset = FakeDataset()

    sampler = data.DistributedSampler(train_dataset, drop_last=True)

    train_dataloader = flow.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler
    )

    logger.print("dataset loaded")

    if flow.env.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader)

    totol_batch = len(train_dataloader)

    # init model and graph
    num_layers = 2
    s0_model = Stage0Model(num_layers, num_hidden[:2], args.patch_size, org_width=args.img_width)
    s1_model = Stage1Model(num_layers, num_hidden[2:], args.patch_size, org_width=args.img_width)
    model = PredRNNPipeline(s0_model, s1_model, args.total_length, args.input_length, num_hidden[-1],
                            args.patch_size, [P0, P1])
    numel = sum([p.numel() for p in model.parameters()])
    logger.print("model size: ", numel)

    sgd = flow.optim.SGD(model.parameters(), lr=0.001)

    graph_pipeline = PredRNNGraph(model, sgd, args, [P0, P1])
    graph_pipeline.debug(1)

    # train
    total_loss = 0
    for epoch in range(1):
        for batch_idx, batch_data in enumerate(train_dataloader, 1):
            batch_data = flow.from_numpy(reshape_patch(batch_data, args.patch_size))
            # batch_data = flow.tensor(batch_data, dtype=flow.float32, placement=P0, sbp=S0)
            # batch_data = reshape_patch(batch_data, args.patch_size)
            batch_data = batch_data.to_global(placement=P0, sbp=S0)
            # batch_data = flow.tensor(batch_data,
            #                          dtype=flow.float32, placement=P0,
            #                          sbp=S0)
            _, mask = schedule_sampling(1.0, epoch)
            mask = flow.from_numpy(mask)
            mask = mask.to_global(placement=P0, sbp=S0)
            # mask = flow.tensor(mask, dtype=flow.float32, placement=P0, sbp=S0)

            loss = graph_pipeline(batch_data, mask)
        #     loss_aver = loss.sum().item() / args.batch_size
        #     total_loss += loss.sum().item()
        #     if batch_idx % args.display_interval == 0:
        #         logger.print(f'epoch: {epoch}, batch: {batch_idx} / {totol_batch}, loss: {total_loss / batch_idx}')
        #         tb.add_scalar('TrainLoss', loss_aver, batch_idx)
        #         tb.add_scalar('TrainLoss2', total_loss / batch_idx, batch_idx)
        #
        # flow.save(model.state_dict(), args.checkpoint_path, global_dst_rank=0)


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

    with flow.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            batch_data = flow.tensor(reshape_patch(batch_data, args.patch_size),
                                     dtype=flow.float32, placement=P0,
                                     sbp=BROADCAST)
            _, mask = schedule_sampling(1.0, 0)
            mask = flow.tensor(mask, dtype=flow.float32, placement=P0, sbp=BROADCAST)
            output = model(batch_data, mask)
            output = output.detach().cpu().to_local()
            batch_data = batch_data.detach().cpu().to_local()
            # output_image(output, args.patch_size, args.output_path)
            loss_recoder.calc_scores(output, batch_data[:, 1:])

    loss_recoder.record()
    loss_recoder.save(args.stats_path)


if __name__ == '__main__':
    if args.run_mode == 'train':
        train_graph()
    elif args.run_mode == 'test':
        evaluate_model()
