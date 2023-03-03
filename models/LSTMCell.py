import oneflow as flow
import oneflow.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden: int, width: int, filter_size, stride):
        super(LSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, groups=1),
            # nn.LayerNorm((num_hidden * 4, width, width))
            nn.LayerNorm(flow.Size([num_hidden * 4, int(width), int(width)]))
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, groups=1),
            # nn.LayerNorm([num_hidden * 4, width, width])
            nn.LayerNorm(flow.Size([num_hidden * 4, int(width), int(width)]))
        )

    def forward(self, x_t, h_t, c_t):
        # print(f"x_t: {x_t.placement, x_t.sbp}")
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, o_x = flow.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = flow.split(h_concat, self.num_hidden, dim=1)

        i_t = flow.sigmoid(i_x + i_h)
        f_t = flow.sigmoid(f_x + f_h + self._forget_bias)
        g_t = flow.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = flow.sigmoid(o_x + o_h + c_new)
        h_new = o_t * flow.tanh(c_new)

        return h_new, c_new









