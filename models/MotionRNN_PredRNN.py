"""
MotionRNN+PredRNN
单机实现
"""

import oneflow as flow
import oneflow.nn as nn
from oneflow.nn import init
from models.SpatioTemporalLSTMCell_Motion_Highway import SpatioTemporalLSTMCell
from models.MotionGRU import MotionGRU


class MotionRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super().__init__()
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour

        cell_list = []
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                                       configs.filter_size, configs.stride, configs.layer_norm),
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),  # - 1
            )
        motion_list = []
        for i in range(num_layers - 1):
            motion_list.append(
                MotionGRU(num_hidden[i] // 4, self.motion_hidden, self.neighbour)
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2,  # + 1
                                   padding=1),
            )
        gate_list = []
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(num_hidden[i] * 2, num_hidden[i], kernel_size=configs.filter_size, stride=1,
                          padding=configs.filter_size // 2),
            )
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch, 1, stride=1, padding=0, bias=False)
        self.conv_first_v = nn.Conv2d(self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False)

    def forward(self, frames, mask_true):
        # [batch, length, channel, height, width]
        output = []
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        for i in range(self.num_layers):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i], self.patch_height, self.patch_width])\
                .to(frames.device)
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i] // 4, self.patch_height // 2,
                 self.patch_width // 2])\
                .to(frames.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = flow.empty(
                [self.configs.batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2])\
                .to(frames.device)
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)

        mem = flow.empty([self.configs.batch_size, self.num_hidden[0], self.patch_height, self.patch_width])\
            .to(frames.device)
        # motion_highway = flow.empty(
        #     [self.configs.batch_size, self.num_hidden[0], self.patch_height, self.patch_width])\
        #     .to(frames.device)
        nn.init.xavier_normal_(mem)
        # nn.init.xavier_normal_(motion_highway)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                frame = frames[:, t]
            else:
                frame = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                      (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            motion_highway = self.conv_first_v(frame)
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](frame, h_t[0], c_t[0], mem, motion_highway)
            frame = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](frame, h_t_conv_offset[0], mean[0])
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = flow.sigmoid(self.gate_list[0](flow.cat([h_t_tmp, h_t[0]], dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            for i in range(1, self.num_layers - 1):
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], mem, motion_highway)
                frame = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](frame, h_t_conv_offset[i], mean[i])
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = flow.sigmoid(self.gate_list[i](flow.cat([h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]

            h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway = self.cell_list[
                self.num_layers - 1](
                h_t[self.num_layers - 2], h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem, motion_highway)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            output.append(x_gen)

        output = flow.stack(output, dim=1)

        return output


class MotionRNNGraph(nn.Graph):
    def __init__(self, model, sgd, configs):
        super().__init__()
        self.model = model
        self.loss_fn = flow.nn.MSELoss().to("cuda")
        self.add_optimizer(sgd)

        # if configs.grad_acc > 1:
        #     self.config.set_gradient_accumulation_steps(configs.grad_acc)
        if configs.amp:
            self.config.enable_amp(True)

    def build(self, inputs, mask):
        out = self.model(inputs, mask)
        loss = self.loss_fn(out, inputs[:, 1:])
        loss.backward()
        return loss
