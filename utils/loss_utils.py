import json
from datetime import datetime

import oneflow as flow
import oneflow.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_PSNR(prediction, truth, data_range=1):
    """Peak Signal Noise Ratio
    最小值为0，值越大越好
    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray
    data_range : int
    Returns
    -------
    ret : np.ndarray
    """
    mse = np.square(prediction - truth).mean(axis=(2, 3, 4))
    ret = 10.0 * np.log10(data_range ** 2 / mse)
    ret = ret.sum(axis=1)
    return ret


def range_map(data, MIN, MAX):
    """
    归一化映射到任意区间
    """
    d_min = np.min(data)  # 当前数据最大值
    d_max = np.max(data)  # 当前数据最小值
    return MIN + (MAX - MIN) / (d_max - d_min) * (data - d_min)


def get_loss_stat(pred, truth):
    """
    [B, S, C, H, W]
    :param pred:
    :param truth:
    :return:
    """
    mse_criterion = flow.nn.MSELoss()
    mae_criterion = flow.nn.L1Loss()
    mse_score = mse_criterion(pred, truth)
    mae_score = mae_criterion(pred, truth)

    pred_numpy = pred.numpy()
    truth_numpy = truth.numpy()
    batch_size = pred.shape[0]
    seq_len = pred.shape[1]
    ssim_score = 0
    # psnr_score = 0
    for i in range(batch_size):
        for j in range(seq_len):
            ssim_score += structural_similarity(pred_numpy[i][j], truth_numpy[i][j], win_size=11, channel_axis=0)
            # img1 = range_map(pred_numpy[i][j], 0, 255)
            # img2 = range_map(truth_numpy[i][j], 0, 255)
            # psnr_score += peak_signal_noise_ratio(img1, img2, data_range=255)
    psnr_score = get_PSNR(range_map(pred_numpy, 0, 1), range_map(truth_numpy, 0, 1), 1).sum()

    return mse_score, mae_score, ssim_score, psnr_score


_GLOBAL_LOSS_RECODER = None


def get_loss_recoder(rank, print_ranks):
    global _GLOBAL_LOSS_RECODER
    if _GLOBAL_LOSS_RECODER is None:
        _GLOBAL_LOSS_RECODER = LossRecoder(rank, print_ranks)

    return _GLOBAL_LOSS_RECODER


class LossRecoder(object):

    def __init__(self, rank, print_ranks, **kwargs):
        self.rank = rank
        self.print_ranks = print_ranks
        self.score_list = np.array([0, 0, 0, 0], dtype='float64')

        self.stats = {
            'mse_score': 0,
            'mae_score': 0,
            'ssim_score': 0,
            'psnr_score': 0,
        }

        self.stats.update(kwargs)

    def check_rank(self):
        return self.rank in self.print_ranks

    def calc_scores(self, pred, truth):
        if not self.check_rank():
            return

        mse, mae, ssim, psnr = get_loss_stat(pred, truth)
        print(mse, mae, ssim, psnr)
        self.score_list += [mse, mae, ssim, psnr]

    def get_calc_scores(self, pred, truth):
        if not self.check_rank():
            return

        mse, mae, ssim, psnr = get_loss_stat(pred, truth)
        return mse, mae, ssim, psnr

    def record(self):
        if not self.check_rank():
            return

        tmp = {}
        for idx, key in enumerate(self.stats.keys()):
            if idx >= 4:
                break
            self.stats[key] = np.round(self.score_list[idx], 6)
            tmp[key + '_per_batch'] = np.round(self.score_list[idx] / self.stats['batch_num'], 4)

        self.stats.update(tmp)
        self.stats['test_time'] = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # 记录时间

    def save(self, path):
        if not self.check_rank():
            return

        with open(path, 'a+') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
            json.dump(self.stats, f, indent=4)
            f.write('\n\n')
