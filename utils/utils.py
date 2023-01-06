import oneflow
from oneflow import nn
from collections import OrderedDict
import numpy as np
import argparse


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:  #
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    num_channels = np.shape(img_tensor)[2]
    img_height = np.shape(img_tensor)[3]
    img_width = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [batch_size, seq_length,
                                num_channels,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                ])
    b = np.transpose(a, [0, 1, 2, 4, 6, 3, 5])
    patch_tensor = np.reshape(b, [batch_size, seq_length,
                                  patch_size * patch_size * num_channels,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  ])
    return patch_tensor


def get_tflops(model_numel, batch_size, seq_len, step_time):
    """
    计算每秒浮点运算多少万亿次
    """
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size')
    parser.add_argument('--patch_size',
                        default=4,
                        type=int,
                        help='image patch size')
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--total_length', type=int, default=20)
    parser.add_argument('--img_height', type=int, default=900)
    parser.add_argument('--img_width', type=int, default=900)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--filter_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--layer_norm', type=int, default=1)
    parser.add_argument('--scheduled_sampling', type=int, default=0)

    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/")
    parser.add_argument('--output_path', type=str, default="output_images/")
    parser.add_argument('--run_mode', type=str, default="train")
    parser.add_argument('--tb_summary_path', type=str, default="runs/")
    parser.add_argument('--stats_path', type=str, default="model_stats/")
    parser.add_argument('--gpus', type=str, default="2,3")

    # 和流水线并行结合的技术
    parser.add_argument('--grad_acc',
                        default=1,
                        type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--amp',
                        default=False,
                        help='enable automatic mixed precision',
                        action='store_true')
    parser.add_argument('--encoder_ac',
                        default=False,
                        help='enable activation checkpointing for encoder part',
                        action='store_true')
    parser.add_argument('--decoder_ac',
                        default=False,
                        help='enable activation checkpointing for decoder part',
                        action='store_true')

    return parser
