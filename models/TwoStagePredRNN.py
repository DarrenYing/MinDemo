"""
PredRNN流水线并行测试
两阶段流水线并行
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import oneflow as flow
from oneflow import nn
from oneflow.nn.graph import GraphModule

from models.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

BROADCAST = [flow.sbp.broadcast]


class Stage0Model(nn.Module):
    def __init__(self, num_layers, num_hidden, patch_size, org_width=900):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.frame_channel = patch_size ** 2
        cell_list = []

        width = org_width // patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, filter_size=5, stride=1)
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

    def forward(self, frame, memory):

        if not len(self.h_t):
            self.init_hc(frame.shape, frame.placement, frame.sbp)

        self.h_t[0], self.c_t[0], memory = self.cell_list[0](frame, self.h_t[0], self.c_t[0], memory)
        for i in range(1, self.num_layers):
            self.h_t[i], self.c_t[i], memory = self.cell_list[i](self.h_t[i - 1], self.h_t[i], self.c_t[i], memory)

        return self.h_t[self.num_layers - 1], memory


class Stage1Model(nn.Module):
    def __init__(self, num_layers, num_hidden, patch_size, org_width=900):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = org_width // patch_size

        for i in range(num_layers):
            in_channel = num_hidden[0] if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, filter_size=5, stride=1)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.h_t = []
        self.c_t = []

        self.conv_last = nn.Conv2d(num_hidden[-1], patch_size ** 2,
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

    def forward(self, h, memory):

        if not len(self.h_t):
            self.init_hc(h.shape, h.placement, h.sbp)

        self.h_t[0], self.c_t[0], memory = self.cell_list[0](h, self.h_t[0], self.c_t[0], memory)
        for i in range(1, self.num_layers):
            self.h_t[i], self.c_t[i], memory = self.cell_list[i](self.h_t[i - 1], self.h_t[i], self.c_t[i], memory)

        output = self.conv_last(self.h_t[self.num_layers - 1])
        # return self.h_t[self.num_layers - 1]

        return output, memory


class PredRNNPipeline(nn.Module):
    def __init__(self, s0_model, s1_model, seq_len, input_len, num_hidden, patch_size, placementCfg):
        super().__init__()
        self.s0_model = s0_model
        self.s1_model = s1_model
        self.seq_len = seq_len
        self.input_len = input_len
        self.num_hidden = num_hidden
        self.out_channel = patch_size ** 2
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

        # init memory
        shape_f = frames.shape
        memory = flow.zeros([shape_f[0], self.num_hidden, shape_f[3], shape_f[4]], dtype=flow.float32,
                            placement=frames.placement,
                            sbp=frames.sbp)

        for t in range(self.seq_len - 1):

            if t < self.input_len:
                frame = frames[:, t]
            else:
                frame = mask_true[:, t - self.input_len] * frames[:, t] + \
                        (1 - mask_true[:, t - self.input_len]) * x_gen

            # conv-lstm layer
            h1, memory = self.s0_model(frame, memory)
            h1 = h1.to_global(placement=self.P1)
            memory = memory.to_global(placement=self.P1)
            x_gen, memory = self.s1_model(h1, memory)

            x_gen = x_gen.to_global(placement=self.P0)
            memory = memory.to_global(placement=self.P0)
            output.append(x_gen)

        output = flow.stack(output, dim=1)
        return output


class PredRNNGraph(nn.Graph):
    def __init__(self, pipeline, sgd, configs, placementCfg):
        super().__init__()
        self.P0 = placementCfg[0]
        self.P1 = placementCfg[1]
        self.module_pipeline = pipeline
        self.module_pipeline.s0_model.to(GraphModule).set_stage(stage_id=0, placement=self.P0)
        self.module_pipeline.s1_model.to(GraphModule).set_stage(stage_id=1, placement=self.P1)
        self.loss_fn = flow.nn.MSELoss()
        self.add_optimizer(sgd)

        if configs.grad_acc > 1:
            self.config.set_gradient_accumulation_steps(configs.grad_acc)
        if configs.amp:
            self.config.enable_amp(True)
        if configs.encoder_ac:
            self.module_pipeline.s0_model.to(GraphModule).activation_checkpointing = True  # 开启激活重计算
        if configs.decoder_ac:
            self.module_pipeline.s1_model.to(GraphModule).activation_checkpointing = True  # 开启激活重计算

    def build(self, inputs, mask):
        out = self.module_pipeline(inputs, mask)
        loss = self.loss_fn(out, inputs[:, 1:])
        loss.backward()
        return loss



