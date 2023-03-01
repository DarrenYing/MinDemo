"""
MotionRNN+PredRNN
单机实现
"""

import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.graph import GraphModule

from models.SpatioTemporalLSTMCell_Motion_Highway import SpatioTemporalLSTMCell
from models.MotionGRUForPipeline import MotionGRU

BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", ranks=[0])
P1 = flow.placement("cuda", ranks=[1])


class Stage0Model(nn.Module):
    """
    conv_first_v + 其他的前两层
    """

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
        for i in range(num_layers):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),
            )
        motion_list = []
        for i in range(num_layers):
            motion_list.append(
                MotionGRU(num_hidden[i] // 4, self.motion_hidden, self.neighbour)
            )
        dec_list = []
        for i in range(num_layers):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2,
                                   padding=1),
            )
        gate_list = []
        for i in range(num_layers):
            gate_list.append(
                nn.Conv2d(num_hidden[i] * 2, num_hidden[i], kernel_size=configs.filter_size, stride=1,
                          padding=configs.filter_size // 2),
            )
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        self.conv_first_v = nn.Conv2d(self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False)

        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

    def reset_hc(self):
        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

    def init_hc(self, input_placement, input_sbp):
        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

        for i in range(self.num_layers):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i], self.patch_height, self.patch_width],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)

            # nn.init.xavier_normal_(zeros)
            self.h_t.append(zeros)
            self.c_t.append(zeros)

        for i in range(self.num_layers):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i] // 4, self.patch_height // 2, self.patch_width // 2],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)

            # nn.init.xavier_normal_(zeros)
            self.h_t_conv.append(zeros)
            zeros = flow.empty(
                [self.configs.batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)
            # nn.init.xavier_normal_(zeros)
            self.h_t_conv_offset.append(zeros)
            self.mean.append(zeros)

    def forward(self, frame, mem):

        if not len(self.h_t):
            self.init_hc(frame.placement, frame.sbp)

        motion_highway = self.conv_first_v(frame)

        self.h_t[0], self.c_t[0], mem, motion_highway = \
            self.cell_list[0](frame, self.h_t[0], self.c_t[0], mem, motion_highway)
        frame = self.enc_list[0](self.h_t[0])
        self.h_t_conv[0], self.h_t_conv_offset[0], self.mean[0] = \
            self.motion_list[0](frame, self.h_t_conv_offset[0], self.mean[0])
        h_t_tmp = self.dec_list[0](self.h_t_conv[0])
        o_t = flow.sigmoid(self.gate_list[0](flow.cat([h_t_tmp, self.h_t[0]], dim=1)))
        self.h_t[0] = o_t * h_t_tmp + (1 - o_t) * self.h_t[0]

        for i in range(1, self.num_layers):
            self.h_t[i], self.c_t[i], mem, motion_highway = \
                self.cell_list[i](self.h_t[i - 1], self.h_t[i], self.c_t[i], mem, motion_highway)
            frame = self.enc_list[i](self.h_t[i])
            self.h_t_conv[i], self.h_t_conv_offset[i], self.mean[i] = \
                self.motion_list[i](frame, self.h_t_conv_offset[i], self.mean[i])
            h_t_tmp = self.dec_list[i](self.h_t_conv[i])
            o_t = flow.sigmoid(self.gate_list[i](flow.cat([h_t_tmp, self.h_t[i]], dim=1)))
            self.h_t[i] = o_t * h_t_tmp + (1 - o_t) * self.h_t[i]

        return self.h_t[self.num_layers - 1], mem, motion_highway


class Stage1Model(nn.Module):
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
            in_channel = num_hidden[0] if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.patch_height, self.patch_width,
                                       configs.filter_size, configs.stride, configs.layer_norm),
            )
        enc_list = []
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(num_hidden[i], num_hidden[i] // 4, kernel_size=configs.filter_size, stride=2,
                          padding=configs.filter_size // 2),
            )
        motion_list = []
        for i in range(num_layers - 1):
            motion_list.append(
                MotionGRU(num_hidden[i] // 4, self.motion_hidden, self.neighbour)
            )
        dec_list = []
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(num_hidden[i] // 4, num_hidden[i], kernel_size=4, stride=2,
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

        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

    def reset_hc(self):
        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

    def init_hc(self, input_placement, input_sbp):
        self.h_t = []
        self.c_t = []
        self.h_t_conv = []
        self.h_t_conv_offset = []
        self.mean = []

        for i in range(self.num_layers):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i], self.patch_height, self.patch_width],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)

            # nn.init.xavier_normal_(zeros)
            self.h_t.append(zeros)
            self.c_t.append(zeros)

        for i in range(self.num_layers - 1):
            zeros = flow.empty(
                [self.configs.batch_size, self.num_hidden[i] // 4, self.patch_height // 2, self.patch_width // 2],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)

            # nn.init.xavier_normal_(zeros)
            self.h_t_conv.append(zeros)
            zeros = flow.empty(
                [self.configs.batch_size, self.motion_hidden, self.patch_height // 2, self.patch_width // 2],
                dtype=flow.float32, placement=input_placement, sbp=input_sbp)
            # nn.init.xavier_normal_(zeros)
            self.h_t_conv_offset.append(zeros)
            self.mean.append(zeros)

    def forward(self, h, mem, motion_highway):

        if not len(self.h_t):
            self.init_hc(h.placement, h.sbp)

        self.h_t[0], self.c_t[0], mem, motion_highway = \
            self.cell_list[0](h, self.h_t[0], self.c_t[0], mem, motion_highway)
        frame = self.enc_list[0](self.h_t[0])
        self.h_t_conv[0], self.h_t_conv_offset[0], self.mean[0] = \
            self.motion_list[0](frame, self.h_t_conv_offset[0], self.mean[0])
        h_t_tmp = self.dec_list[0](self.h_t_conv[0])
        o_t = flow.sigmoid(self.gate_list[0](flow.cat([h_t_tmp, self.h_t[0]], dim=1)))
        self.h_t[0] = o_t * h_t_tmp + (1 - o_t) * self.h_t[0]

        self.h_t[1], self.c_t[1], mem, motion_highway = \
            self.cell_list[1](self.h_t[0], self.h_t[1], self.c_t[1], mem, motion_highway)
        output = self.conv_last(self.h_t[1])

        return output, mem


class MotionRNNPipeline(nn.Module):
    def __init__(self, s0_model, s1_model, num_hidden, configs, placementCfg):
        super().__init__()
        self.s0_model = s0_model
        self.s1_model = s1_model
        self.num_hidden = num_hidden
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
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
        mem = flow.empty([self.configs.batch_size, self.num_hidden, self.patch_height, self.patch_width],
                         dtype=flow.float32, placement=frames.placement, sbp=frames.sbp)
        # motion_highway = flow.empty(
        #                  [self.configs.batch_size, self.num_hidden[0], self.patch_height, self.patch_width],
        #                  dtype=flow.float32, placement=frames.placement, sbp=frames.sbp)
        # nn.init.xavier_normal_(mem)
        # nn.init.xavier_normal_(motion_highway)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                frame = frames[:, t]
            else:
                frame = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                        (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            # conv-lstm layer
            h1, mem, motion_highway = self.s0_model(frame, mem)
            h1 = h1.to_global(placement=self.P1, sbp=frames.sbp)
            mem = mem.to_global(placement=self.P1, sbp=frames.sbp)
            motion_highway = motion_highway.to_global(placement=self.P1, sbp=frames.sbp)

            x_gen, mem = self.s1_model(h1, mem, motion_highway)
            x_gen = x_gen.to_global(placement=self.P0, sbp=frames.sbp)
            mem = mem.to_global(placement=self.P0, sbp=frames.sbp)
            output.append(x_gen)

        output = flow.stack(output, dim=1)
        return output


class MotionRNNGraph(nn.Graph):
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
