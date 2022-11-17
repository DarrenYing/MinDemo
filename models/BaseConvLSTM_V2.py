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


class BaseConvLSTM(nn.Module):
    def __init__(self, num_layers, num_hidden, patch_size,
                 seq_len, input_len, org_width=900):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.seq_len = seq_len
        self.input_len = input_len
        self.frame_channel = patch_size ** 2
        cell_list = []

        width = org_width // patch_size

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                LSTMCell(in_channel, num_hidden[i], width, filter_size=5, stride=1)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask):
        # 图片序列，[batch, length, channel, height, width]
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        output = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = flow.zeros([batch, self.num_hidden[i], height, width], dtype=flow.float32).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.seq_len - 1):

            if t < self.input_len:
                frame = frames[:, t]  # t是length维度
            else:
                frame = mask[:, t - self.input_len] * frames[:, t] + \
                      (1 - mask[:, t - self.input_len]) * x_gen

            # conv-lstm layer
            h_t[0], c_t[0] = self.cell_list[0](frame, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                # print('layer number is:',str(i),memory.shape,h_t[i].shape)
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            output.append(x_gen)

        output = flow.stack(output, dim=1)
        return output


class BasePredRNNGraph(nn.Graph):
    def __init__(self, model, sgd, configs):
        super().__init__()
        self.model = model
        self.loss_fn = flow.nn.MSELoss()
        self.add_optimizer(sgd)

        if configs.grad_acc > 1:
            self.config.set_gradient_accumulation_steps(configs.grad_acc)
        if configs.amp:
            self.config.enable_amp(True)
        if configs.encoder_ac:
            self.module_pipeline.s0_model.config.activation_checkpointing = True  # 开启激活重计算
        if configs.decoder_ac:
            self.module_pipeline.s1_model.config.activation_checkpointing = True  # 开启激活重计算

    def build(self, inputs, mask):
        out = self.model(inputs, mask)
        # logger.print(out.shape)
        loss = self.loss_fn(out, inputs[:, 1:])
        loss.backward()
        return loss


