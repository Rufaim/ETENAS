import os
import sys
import time
import argparse
import matplotlib.pyplot as pyplot
import random
from easydict import EasyDict as edict
import numpy as np
import tqdm
import torch
from torch import nn
from pathlib import Path
from typing import List, Tuple, Callable
from dataclasses import dataclass
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, MetricType, get_linear_region_counter_v2, get_ntk_n, get_ntk_n_v2, \
        get_nngp_n, get_nngp_n_v2, regional_division_counter, RegionDivisionScoreType, synflow, logsynflow, zen_score
from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import SimpleApi as API
from prune_tenas import init_model


@dataclass
class MethodDescriptor(object):
    name: str
    codename: str
    func: Callable


METHODS_LIST = [
    MethodDescriptor("Conditional number NTK", "cond_ntk_v1", lambda loader, _, net: get_ntk_n(loader, [net], recalbn=0, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Accuracy NNGP", "acc_nngp_v1", lambda train_loader, valid_loader, net: get_nngp_n(train_loader, valid_loader, [net], train_mode=True, num_batch=2)[0]),

    MethodDescriptor("Accuracy of NTK", "acc_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("MSE of NTK", "mse_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Label-Gradient Alignment of NTK", "lga_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Frobenius norm of NTK", "fro_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Mean value of NTK", "mean_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Conditional number of NTK", "cond_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, num_batch=1)[0]),
    MethodDescriptor("Eigenvalue score of NTK", "eig_ntk", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, num_batch=1)[0]),

    MethodDescriptor("Accuracy of NTK(correlation)", "acc_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("MSE of NTK(correlation)", "mse_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("Label-Gradient Alignment of NTK(correlation)", "lga_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("Frobenius norm of NTK(correlation)", "fro_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("Mean value of NTK(correlation)", "mean_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("Conditional number of NTK(correlation)", "cond_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, as_correlation=True, num_batch=1)[0]),
    MethodDescriptor("Eigenvalue score of NTK(correlation)", "eig_ntk_corr", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, as_correlation=True, num_batch=1)[0]),

    MethodDescriptor("Accuracy of NNGP", "acc_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP", "mse_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP", "lga_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP", "fro_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP", "mean_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("Conditional number of NNGP", "cond_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP", "eig_nngp", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, num_batch=2)[0]),

    MethodDescriptor("Accuracy of NNGP(correlation)", "acc_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP(correlation)", "mse_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(correlation)", "lga_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP(correlation)", "fro_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP(correlation)", "mean_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("Conditional number of NNGP(correlation)", "cond_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, as_correlation=True, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(correlation)", "eig_nngp_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, as_correlation=True, num_batch=2)[0]),

    MethodDescriptor("Accuracy of NNGP(readout)", "acc_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP(readout)", "mse_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(readout)", "lga_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP(readout)", "fro_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP(readout)", "mean_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Condition number of NNGP(readout)", "cond_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(readout)", "eig_nngp_read", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, as_correlation=False, use_logits=True, num_batch=2)[0]),

    MethodDescriptor("Accuracy of NNGP(readout, correlation)", "acc_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP(readout, correlation)", "mse_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(readout, correlation)", "lga_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP(readout, correlation)", "fro_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP(readout, correlation)", "mean_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Conditional number of NNGP(readout, correlation)", "cond_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(readout, correlation)", "eig_nngp_read_corr", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, as_correlation=True, use_logits=True, num_batch=2)[0]),

    MethodDescriptor("Accuracy of NTK(1 train epoch)", "acc_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("MSE of NTK(1 train epoch)", "mse_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("Label-Gradient Alignment of NTK(1 train epoch)", "lga_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("Frobenius norm of NTK(1 train epoch)", "fro_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("Mean value of NTK(1 train epoch)", "mean_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("Conditional number of NTK(1 train epoch)", "cond_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, train_iters=1, num_batch=1)[0]),
    MethodDescriptor("Eigenvalue score of NTK(1 train epoch)", "eig_ntk_train", lambda train_loader, valid_loader, net: get_ntk_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, train_iters=1, num_batch=1)[0]),

    MethodDescriptor("Accuracy of NNGP(1 train epoch)", "acc_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP(1 train epoch)", "mse_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(1 train epoch)", "lga_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP(1 train epoch)", "fro_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP(1 iterations epoch)", "nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("Conditional number of NNGP(1 train epoch)", "cond_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, train_iters=1, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(1 train epoch)", "eig_nngp_train", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, train_iters=1, num_batch=2)[0]),

    MethodDescriptor("ReLU regions distance", "regs_dist_full", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.FULL, num_batch=1, verbose=False)[0]),
    MethodDescriptor("ReLU regions distance(max over batch)", "regs_dist_max", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.MAX, num_batch=3, verbose=False)[0]),
    MethodDescriptor("ReLU regions distance(mean over batch)", "regs_dist_mean", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.MEAN, num_batch=3, verbose=False)[0]),

    MethodDescriptor("ReLU regions distance(1 train epoch)", "regs_dist_full_train", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.FULL, train_iters=1, num_batch=10, verbose=False)[0]),
    MethodDescriptor("ReLU regions distance(max over batch, 1 train epoch)", "regs_dist_max_train", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.MAX, train_iters=1, num_batch=10, verbose=False)[0]),
    MethodDescriptor("ReLU regions distance(mean over batch, 1 train epoch)", "regs_dist_mean_train", lambda train_loader, _, net: regional_division_counter(train_loader, [net], train_mode=False, score_type=RegionDivisionScoreType.MEAN, train_iters=1, num_batch=10, verbose=False)[0]),

    MethodDescriptor("Accuracy of NNGP(more batches)", "acc_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("MSE of NNGP(more batches)", "mse_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(more batches)", "lga_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("Frobenius norm of NNGP(more batches)", "fro_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("Mean value of NNGP(more batches)", "mean_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("Conditional number of NNGP(more batches)", "cond_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, num_batch=40)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(more batches)", "eig_nngp_bb", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, num_batch=40)[0]),

    MethodDescriptor("Expected number of ReLU regions", "regs_num", lambda train_loader, _, net: get_linear_region_counter_v2(train_loader, [net], train_mode=False, num_batch=3)[0]),

    MethodDescriptor("Accuracy of NNGP(20 train iterations )", "acc_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.ACC, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("MSE of NNGP(20 train iterations )", "mse_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MSE, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("Label-Gradient Alignment of NNGP(20 train iterations )", "lga_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.LGA, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("Frobenius norm of NNGP(20 train iterations )", "fro_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.FRO, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("Mean value of NNGP(20 iterations)", "mean_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.MEAN, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("Conditional number of NNGP(20 train iterations )", "cond_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.COND, train_mode=True, train_iters=20, num_batch=2)[0]),
    MethodDescriptor("Eigenvalue score of NNGP(20 train iterations )", "eig_nngp_train_it", lambda train_loader, valid_loader, net: get_nngp_n_v2(train_loader, valid_loader, [net], metric=MetricType.EIG, train_mode=True, train_iters=20, num_batch=2)[0]),

    MethodDescriptor("SynFlow", "synflow", lambda train_loader, _, net: synflow(train_loader, [net], train_mode=True)[0]),
    MethodDescriptor("LogSynFlow", "logsynflow", lambda train_loader, _, net: logsynflow(train_loader, [net], train_mode=True)[0]),
    MethodDescriptor("Zen-Score", "zen_score", lambda train_loader, _, net: zen_score(train_loader, [net], train_mode=False)[0])
]


class RandomArchSampler(object):
    INFINITY = -1000
    def __init__(self, shapes: List[Tuple[int,...]]):
        self.shapes = shapes

    def sample(self):
        arch_parameters = []
        for sh in self.shapes:
            alpha = torch.rand(sh)
            alpha[:, 0] = RandomArchSampler.INFINITY
            alpha = torch.argmax(alpha, dim=1).long()
            alpha = nn.functional.one_hot(alpha, sh[1])
            arch_parameters.append(alpha)
        return arch_parameters


def is_expected_regions_number(method: MethodDescriptor):
    if method == "regs_num":
        return True
    return False


def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(xargs.save_path, "raw_data"), exist_ok=True)

    # choosing test method
    for m in METHODS_LIST:
        if m.codename == args.method:
            method = m
            print("testing {} on {}".format(m.name, xargs.dataset))
            break
    else:
        print(f"No such method as {args.method}")
        exit(1)

    batch_size = args.batch_size
    input_size = None
    if is_expected_regions_number(args.method):
        input_size = (batch_size, 1, 3, 3)
    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, input_size=input_size, cutout=-1)
    _, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/dataset_split/', batch_size, xargs.workers)

    search_space = get_search_spaces('cell', 'nas-bench-201')
    model_config = edict({'name': 'DARTS-V1',
                          'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                          'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                          'space': search_space,
                          'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                          })
    if is_expected_regions_number(args.method):
        model_config = edict({'name': 'DARTS-V1',
                                   'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                   'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                   'space': search_space,
                                   'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                                   })
    network = get_cell_based_tiny_net(model_config).cuda().train()
    init_model(network, xargs.init)
    sampler = RandomArchSampler([a.shape for a in network.get_alphas()])

    api = API(xargs.arch_nas_dataset)
    values = []
    accuracies = []
    genotypes = []
    times = []
    for _ in tqdm.trange(xargs.num_samples):
        alphas = sampler.sample()
        network.set_alphas(alphas)
        genotype = network.genotype()

        start_time = time.time()
        value = 0
        for _ in range(xargs.repeat):
            # random reinit
            init_model(network, xargs.init + "_fanout" if xargs.init.startswith('kaiming') else xargs.init)
            value += method.func(train_loader, valid_loader, network)
        value /= xargs.repeat
        times.append((time.time() - start_time) / xargs.repeat)

        idx = api.query_index_by_arch(genotype)
        acc = api.query_by_index(idx)[xargs.dataset]

        values.append(value)
        accuracies.append(acc)
        genotypes.append(genotype)

    time_per_iteration = np.mean(times)
    print(f"Time per iteration: {time_per_iteration} sec")

    ## making plot

    fig = pyplot.figure()
    pyplot.scatter(values, accuracies, s=5.5, c="b", marker="o")
    pyplot.xlabel(method.name, fontsize=15)
    pyplot.ylabel(f"Test accuracy ({xargs.dataset.upper()})", fontsize=15)
    pyplot.savefig(os.path.join(xargs.save_path, f"{method.codename}_{xargs.dataset}.png"))
    pyplot.close(fig)

    # saving results
    np.savez(os.path.join(xargs.save_path, "raw_data", f"{method.codename}_{xargs.dataset}.npz"),
             metric=values,
             accuracy=accuracies,
             genotypes=genotypes,
             time_per_iteration=time_per_iteration)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Metric plotter for NAS-bench-201")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1], help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')
    parser.add_argument('--save_path', type=str, help='Folder to save plots')
    parser.add_argument('--arch_nas_dataset', type=str, help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--precision', type=int, default=3, help='precision for % of changes of cond(NTK) and #Regions')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of NTK and Regions')
    parser.add_argument('--init', default='kaiming_norm', choices=['kaiming_norm', 'kaiming_norm_fanin', 'kaiming_norm_fanout'], help='choose init')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples for the graph')
    parser.add_argument('--method', type=str, choices=[m.codename for m in METHODS_LIST], default="cond_ntk", help="choose method to test")
    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
