"""
PredRNN形式的ConvLSTM流水线并行测试
两阶段流水线并行
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import oneflow as flow
from oneflow import nn

from models.LSTMCell import LSTMCell

BROADCAST = flow.sbp.broadcast


class Stage0Model(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.frame_channel = configs.patch_size ** 2
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                LSTMCell(in_channel, num_hidden[i], width,
                         filter_size=configs.filter_size, stride=configs.stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.h_t = []
        self.c_t = []

    def reset_hc(self):
        self.h_t = []
        self.c_t = []

    def init_hc(self, input_shape, input_placement, input_sbp):
        self.h_t = []
        self.c_t = []
        batch = input_shape[0]
        height = input_shape[2]
        width = input_shape[3]

        for i in range(self.num_layers):
            zeros = flow.zeros([batch, self.num_hidden[i], height, width], dtype=flow.float32,
                               placement=input_placement,
                               sbp=input_sbp
                               )
            self.h_t.append(zeros)
            self.c_t.append(zeros)

    def forward(self, frame):

        if not len(self.h_t):
            self.init_hc(frame.shape, frame.placement, frame.sbp)

        self.h_t[0], self.c_t[0] = self.cell_list[0](frame, self.h_t[0], self.c_t[0])
        for i in range(1, self.num_layers):
            self.h_t[i], self.c_t[i] = self.cell_list[i](self.h_t[i - 1], self.h_t[i], self.c_t[i])

        return self.h_t[self.num_layers - 1]


class Stage1Model(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size

        for i in range(num_layers):
            in_channel = num_hidden[0] if i == 0 else num_hidden[i - 1]
            cell_list.append(
                LSTMCell(in_channel, num_hidden[i], width,
                         filter_size=configs.filter_size, stride=configs.stride)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.h_t = []
        self.c_t = []

        self.conv_last = nn.Conv2d(num_hidden[-1], configs.patch_size ** 2,
                                   kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def reset_hc(self):
        self.h_t = []
        self.c_t = []

    def init_hc(self, input_shape, input_placement, input_sbp):
        self.h_t = []
        self.c_t = []
        batch = input_shape[0]
        height = input_shape[2]
        width = input_shape[3]

        for i in range(self.num_layers):
            zeros = flow.zeros([batch, self.num_hidden[i], height, width], dtype=flow.float32,
                               placement=input_placement,
                               sbp=input_sbp
                               )
            self.h_t.append(zeros)
            self.c_t.append(zeros)

    def forward(self, h):

        if not len(self.h_t):
            self.init_hc(h.shape, h.placement, h.sbp)

        self.h_t[0], self.c_t[0] = self.cell_list[0](h, self.h_t[0], self.c_t[0])
        for i in range(1, self.num_layers):
            self.h_t[i], self.c_t[i] = self.cell_list[i](self.h_t[i - 1], self.h_t[i], self.c_t[i])

        # return self.h_t[self.num_layers - 1]
        output = self.conv_last(self.h_t[self.num_layers - 1])
        return output


class ConvLSTMPipeline(nn.Module):
    def __init__(self, s0_model, s1_model, configs, placementCfg):
        super().__init__()
        self.s0_model = s0_model
        self.s1_model = s1_model
        self.seq_len = configs.total_length
        self.input_len = configs.input_length
        self.out_channel = configs.patch_size ** 2
        self.P0 = placementCfg[0]
        self.P1 = placementCfg[1]

        self.s0_model.to_global(placement=self.P0, sbp=BROADCAST)
        self.s1_model.to_global(placement=self.P1, sbp=BROADCAST)

    def forward(self, frames, mask_true):
        """
        :param frames: [batch, length, channel, height, width]
        :param mask_true: [batch, length, channel, height, width]
        :return:
        """

        output = []

        self.s0_model.reset_hc()
        self.s1_model.reset_hc()

        for t in range(self.seq_len - 1):

            frame = frames[:, t]
            # if t < self.input_len:
            #     frame = frames[:, t]
            # else:
            #     frame = mask_true[:, t - self.input_len] * frames[:, t] + \
            #             (1 - mask_true[:, t - self.input_len]) * x_gen

            # conv-lstm layer
            h1 = self.s0_model(frame)
            h1 = h1.to_global(placement=self.P1)
            x_gen = self.s1_model(h1)
            # 移动到P0
            x_gen = x_gen.to_global(placement=self.P0)
            output.append(x_gen)

        output = flow.stack(output, dim=1)
        return output


class ConvLSTMGraph(nn.Graph):
    def __init__(self, pipeline, sgd, configs, placementCfg):
        super().__init__()
        self.P0 = placementCfg[0]
        self.P1 = placementCfg[1]
        self.module_pipeline = pipeline
        self.module_pipeline.s0_model.config.set_stage(stage_id=0, placement=self.P0)
        self.module_pipeline.s1_model.config.set_stage(stage_id=1, placement=self.P1)
        self.loss_fn = flow.nn.MSELoss()
        self.add_optimizer(sgd)

    def build(self, inputs, mask):
        out = self.module_pipeline(inputs, mask)
        loss = self.loss_fn(out, inputs[:, 1:])
        loss.backward()
        return loss
