import os, sys, time, argparse
import json
import random
from easydict import EasyDict as edict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple
from collections import defaultdict

lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import get_datasets, get_nas_search_loaders
from procedures import prepare_seed, prepare_logger, MetricType, get_linear_region_counter_v2, get_nngp_n_v2, \
    regional_division_counter, RegionDivisionScoreType, zen_score
from utils import get_model_infos, init_model, round_to, is_single_path
from log_utils import time_string
from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import SimpleApi as API

INF = 1000  # used to mark prunned operators


class Ranker(Enum):
    NNGP_frob = "Frobenius norm of NNGP"
    NNGP_eig = "Eigenvalue score of NNGP"
    NNGP_mean = "Mean value of NNGP"
    NNGP_LGA = "Label-Gradient Alignment of NNGP"
    NNGP_cond = "Condition number of NNGP"
    NNGP = "NNGP"
    REGS_num = "Expected number of ReLU regions"
    REGS_dist = "ReLU regions distance"
    ZEN = "Zen-Score"

    def __str__(self):
        return self.value

    def is_override_batch(self):
        if self is Ranker.REGS_num:
            return True
        if self is Ranker.REGS_dist:
            return True
        return False

    def is_nngp_based(self):
        if self.NNGP:
            return True
        if self.NNGP_frob:
            return True
        if self.NNGP_eig:
            return True
        if self.NNGP_mean:
            return True
        if self.NNGP_LGA:
            return True
        if self.NNGP_cond:
            return True
        return False

    def prune_allowed_args(self, args: Dict[str, Any]):
        output_args = {"sign": args.get("sign", 1)}
        if self in [Ranker.NNGP_frob, Ranker.NNGP_eig, Ranker.NNGP_mean, Ranker.NNGP_LGA, Ranker.NNGP_cond]:
            output_args["num_batch"] = args.get("num_batch", 2)
            output_args["train_mode"] = args.get("train_mode", True)
        if self in [Ranker.REGS_num, Ranker.REGS_dist]:
            output_args["num_batch"] = 1
            output_args["train_mode"] = args.get("train_mode", False)
        if self is Ranker.REGS_num:
            output_args["batch"] = args.get("batch", 1000)
        if self is Ranker.REGS_dist:
            output_args["batch"] = args.get("batch", 150)
        if self is Ranker.ZEN:
            output_args["train_mode"] = args.get("train_mode", False)
        return output_args


def parse_rankers_config(path):
    with open(path, "r") as file:
        config = json.load(file)

    rankers = []
    for r in config:
        r_type = Ranker(r["type"])
        kwargs = Ranker.prune_allowed_args(r_type, r.get("args", {}))
        if r_type is Ranker.NNGP_frob:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.FRO,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=True, use_logits=True,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.NNGP_frob:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.FRO,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=True, use_logits=True,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.NNGP_eig:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.EIG,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=False, use_logits=False,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.NNGP_mean:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.MEAN,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=True, use_logits=True,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.NNGP_LGA:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.LGA,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=False, use_logits=True,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.NNGP_cond:
            func = lambda train_loader, valid_loader, nets: get_nngp_n_v2(train_loader, valid_loader, nets,
                                                                          metric=MetricType.COND,
                                                                          train_mode=kwargs["train_mode"],
                                                                          as_correlation=False, use_logits=True,
                                                                          num_batch=kwargs["num_batch"])
        elif r_type is Ranker.REGS_num:
            func = lambda train_loader, _, nets: get_linear_region_counter_v2(train_loader, nets,
                                                                              train_mode=kwargs["train_mode"],
                                                                              num_batch=kwargs["num_batch"])
        elif r_type is Ranker.REGS_dist:
            func = lambda train_loader, _, nets: regional_division_counter(train_loader, nets,
                                                                           train_mode=kwargs["train_mode"],
                                                                           score_type=RegionDivisionScoreType.FULL,
                                                                           num_batch=kwargs["num_batch"], verbose=False)
        elif r_type is Ranker.ZEN:
            func = lambda train_loader, _, nets: zen_score(train_loader, nets, train_mode=kwargs["train_mode"])
        else:
            raise RuntimeError("invalid ranker")
        rankers.append((r_type, func, kwargs))

    return rankers


def prune_func_rank(rankers_list, arch_parameters, model_config, special_model_configs, train_loader, valid_loader,
                    special_dataloaders,
                    edge_groups=None, init="kaiming_norm", repeat=1, prune_number=1):
    for alpha in arch_parameters:
        alpha[:, 0] = -INF

    # set neural networks
    network_origin = get_cell_based_tiny_net(model_config).cuda().train()
    # init_model(network_origin, init)
    network_origin.set_alphas(arch_parameters)
    special_networks_origins = {}
    for r, c in special_model_configs.items():
        special_networks_origins[r] = get_cell_based_tiny_net(c).cuda().train()
        # init_model(special_networks_origins[r], init)
        special_networks_origins[r].set_alphas(arch_parameters)

    alpha_active = [(nn.functional.softmax(alpha, 1) > 0.01).float() for alpha in arch_parameters]
    if edge_groups is None:
        prune_number = min(prune_number,
                           alpha_active[0][0].sum() - 1)  # adjust prune_number based on current remaining ops on each edge
        active_parameter_indexes = [(idx_ct, idx_edge, idx_op)
                                    for idx_ct in range(len(arch_parameters))
                                    for idx_edge in range(arch_parameters[idx_ct].shape[0])
                                    for idx_op in range(arch_parameters[idx_ct].shape[1])
                                    if alpha_active[idx_ct][idx_edge].sum() != 1  # more than one op remaining
                                    if alpha_active[idx_ct][idx_edge, idx_op] > 0  # op is active
                                    ]
    else:
        active_parameter_indexes = [(idx_ct, idx_edge, idx_op)
                                    for idx_ct in range(len(arch_parameters))
                                    for group_st, group_end in edge_groups
                                    for idx_edge in range(group_st, group_end)
                                    for idx_op in range(arch_parameters[idx_ct].shape[1])
                                    if group_end - group_st > prune_number
                                    if alpha_active[idx_ct][idx_edge, idx_op] > 0  # op is active
                                    ]

    edge_scores = []
    for edge_indexes in tqdm(active_parameter_indexes):
        (idx_ct, idx_edge, idx_op) = edge_indexes
        _arch_param = [alpha.detach().clone() for alpha in arch_parameters]
        _arch_param[idx_ct][idx_edge, idx_op] = -INF

        network = get_cell_based_tiny_net(model_config).cuda().train()
        special_networks = {r: get_cell_based_tiny_net(special_model_configs[r]).cuda().train()
                            for r in special_model_configs
                            }
        _scores = []
        for _ in range(repeat):
            ranker_scores = []
            for ranker, rank_func, kwargs in rankers_list:
                ### initializing networks for backward
                init_model(network_origin, init + "_fanout" if init.startswith('kaiming') else init)
                # make sure network_origin and network are identical
                for param_ori, param in zip(network_origin.parameters(), network.parameters()):
                    param.data.copy_(param_ori.data)
                network.set_alphas(_arch_param)

                for k in special_networks_origins:
                    init_model(special_networks_origins[k], init + "_fanout" if init.startswith('kaiming') else init)
                    # make sure network_thin and network_thin_origin are identical
                    for param_ori, param in zip(special_networks_origins[k].parameters(),
                                                special_networks[k].parameters()):
                        param.data.copy_(param_ori.data)
                    special_networks[k].set_alphas(_arch_param)


                origin_net = special_networks_origins.get(ranker, network_origin)
                net = special_networks.get(ranker, network)

                train_loader_, valid_loader_ = special_dataloaders.get(ranker, (train_loader, valid_loader))
                origin_rank, rank = rank_func(train_loader_, valid_loader_, [origin_net, net])

                percent_shift = kwargs["sign"] * (origin_rank - rank) / origin_rank
                ranker_scores.append(percent_shift)
            _scores.append(np.sum(ranker_scores))
        edge_scores.append((edge_indexes, np.mean(_scores)))

    edge_scores = sorted(edge_scores,
                         key=lambda tup: tup[1])  # list of (cell_idx, edge_idx, op_idx), [ntk_rank, regions_rank]

    edge2choice = defaultdict(list)
    for (cell_idx, edge_idx, op_idx), _ in edge_scores:
        if edge_groups is None:
            if len(edge2choice[(cell_idx, edge_idx)]) < prune_number:
                edge2choice[(cell_idx, edge_idx)].append((cell_idx, edge_idx, op_idx))
        else:
            for group_st, group_end in edge_groups:
                if group_st <= edge_idx < group_end and len(edge2choice[(group_st, group_end)]) < (group_end - group_st - prune_number):
                    edge2choice[(group_st, group_end)].append((cell_idx, edge_idx, op_idx))
    for choices in edge2choice.values():
        for (cell_idx, edge_idx, op_idx) in choices:
            arch_parameters[cell_idx].data[edge_idx, op_idx] = -INF

    return arch_parameters

def main(xargs):
    PID = os.getpid()
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    prepare_seed(xargs.rand_seed)

    if xargs.timestamp == 'none':
        xargs.timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))

    xargs.save_dir = os.path.join(xargs.save_dir,
                "repeat{}-prunNum{}-{}-batch{}".format(xargs.repeat, xargs.prune_number, xargs.init, xargs.batch_size),
                str(xargs.timestamp),
                "seed{}".format(xargs.rand_seed))

    # checking ranking list
    rankers_list = parse_rankers_config(xargs.rankers_config)
    batch_size = xargs.batch_size

    train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, cutout=-1)
    _, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                           'configs/dataset_split/', batch_size, xargs.workers)

    special_dataloaders = {}
    for r, _, kwargs in rankers_list:
        if Ranker.is_override_batch(r):
            if kwargs["batch"] != batch_size:
                input_size = None
                if r is Ranker.REGS_num:
                    input_size = (kwargs["batch"], 1, 3, 3)
                train_data_, valid_data_, _, _ = get_datasets(xargs.dataset, xargs.data_path,
                                                            input_size=input_size, cutout=-1)
                _, train_loader_, valid_loader_ = get_nas_search_loaders(train_data_, valid_data_, xargs.dataset,
                                                                       'configs/dataset_split/', kwargs["batch"],
                                                                       xargs.workers)
                special_dataloaders[r] = (train_loader_, valid_loader_)

    ##### config & logging #####
    logger = prepare_logger(xargs)

    logger.log(f'Batch size : {xargs.batch_size}')
    logger.log(f'Input image shape : {xshape[1:]}')
    logger.log(f'Saving dir : {xargs.save_dir}')
    logger.log(f'Rankers :')
    for r, _, args in rankers_list:
        logger.log(f'\t{r.value}: {args}')
    logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}'.format(xargs.dataset, len(train_loader)))
    ###############

    search_space = get_search_spaces('cell', xargs.search_space_name)
    special_model_configs = {}
    if xargs.search_space_name == 'nas-bench-201':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 3, 'N': 1, 'depth': -1, 'use_stem': True,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              })
        if Ranker.REGS_num in special_dataloaders:
            special_model_configs[Ranker.REGS_num] = edict({'name': 'DARTS-V1',
                                                            'C': 1, 'N': 1, 'depth': 1, 'use_stem': False,
                                                            'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                                            'space': search_space,
                                                            'affine': True,
                                                            'track_running_stats': bool(xargs.track_running_stats),
                                                            })
    elif xargs.search_space_name == 'darts':
        model_config = edict({'name': 'DARTS-V1',
                              'C': 1, 'N': 1, 'depth': 2, 'use_stem': True, 'stem_multiplier': 1,
                              'num_classes': class_num,
                              'space': search_space,
                              'affine': True, 'track_running_stats': bool(xargs.track_running_stats),
                              'super_type': xargs.super_type,
                              'steps': 4,
                              'multiplier': 4,
                              })
        if Ranker.REGS_num in special_dataloaders:
            special_model_configs[Ranker.REGS_num] = edict({'name': 'DARTS-V1',
                                                            'C': 1, 'N': 1, 'depth': 2, 'use_stem': False,
                                                            'stem_multiplier': 1,
                                                            'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                                            'space': search_space,
                                                            'affine': True,
                                                            'track_running_stats': bool(xargs.track_running_stats),
                                                            'super_type': xargs.super_type,
                                                            'steps': 4,
                                                            'multiplier': 4,
                                                            })
    else:
        raise RuntimeError(f"{xargs.search_space_name} is not a valid search space")

    logger.log('model-config : {:}'.format(model_config))
    logger.log('special-model-configs : {:}'.format(special_model_configs))

    network = get_cell_based_tiny_net(model_config)

    # ### all params trainable (except train_bn) #########################
    flop, param = get_model_infos(network, xshape)
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))

    network = network.cuda()
    genotypes = {-1: network.genotype()}
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, 1:] = 0
        alpha[:, 0] = -INF

    start_time = time.time()
    epoch = -1
    while not is_single_path(network):
        epoch += 1
        torch.cuda.empty_cache()
        print("<< ============== JOB (PID = %d) %s ============== >>" % (PID, '/'.join(xargs.save_dir.split("/")[-6:])))

        arch_parameters = prune_func_rank(rankers_list, arch_parameters, model_config, special_model_configs,
                                          train_loader, valid_loader, special_dataloaders,
                                          init=xargs.init,
                                          repeat=xargs.repeat,
                                          prune_number=xargs.prune_number
                                          )
        # rebuild supernet
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)
        genotypes[epoch] = network.genotype()

        logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format(
            '\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    if xargs.search_space_name == 'darts':
        print("===>>> Prune Edge Groups...")
        arch_parameters = prune_func_rank(rankers_list, arch_parameters, model_config, special_model_configs,
                                          train_loader, valid_loader, special_dataloaders,
                                          init=xargs.init,
                                          repeat=xargs.repeat,
                                          edge_groups=[(0, 2), (2, 5), (5, 9), (9, 14)],
                                          prune_number=2
                                          )
        network = get_cell_based_tiny_net(model_config)
        network = network.cuda()
        network.set_alphas(arch_parameters)

    logger.log('<<<--->>> End: {:}'.format(network.genotype()))
    logger.log('operators remaining (1s) and prunned (0s)\n{:}'.format(
        '\n'.join([str((alpha > -INF).int()) for alpha in network.get_alphas()])))

    end_time = time.time()
    logger.log('\n' + '-' * 100)
    logger.log(f"Time spent: {end_time - start_time} sec")

    # write final parameters into file
    arch_parameters_npy = [[alpha.detach().clone().cpu().numpy() for alpha in arch_parameters]]
    np.save(os.path.join(xargs.save_dir, "arch_parameters_history.npy"), arch_parameters_npy)

    logger.log(genotypes[epoch] if type(genotypes[epoch]) is str else genotypes[epoch].tostr())

    # check the performance from the architecture dataset (for NAS-Bench-201)
    if xargs.arch_nas_dataset is not None and xargs.search_space_name != 'darts':
        api = API(xargs.arch_nas_dataset)
        logger.log('{:} create API = {:} done'.format(time_string(), api))
        idx = api.query_index_by_arch(genotypes[epoch])
        acc = api.query_by_index(idx)[xargs.dataset]
        logger.log('Test Accuracy {} on {}'.format(acc, xargs.dataset))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ETENAS")
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--rankers_config', type=str, help='path to rankers config')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
    parser.add_argument('--search_space_name', type=str, default='nas-bench-201',
                        help='space of operator candidates: nas-bench-201 or darts.')
    parser.add_argument('--max_nodes', type=int, help='The maximum number of nodes.')
    parser.add_argument('--track_running_stats', type=int, choices=[0, 1],
                        help='Whether use track_running_stats or not in the BN layer.')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for ntk')
    parser.add_argument('--save_dir', type=str, help='Folder to save checkpoints and log.')
    parser.add_argument('--arch_nas_dataset', type=str,
                        help='The path to load the nas-bench-201 architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    parser.add_argument('--prune_number', type=int, default=1,
                        help='number of operator to prune on each edge per round')
    parser.add_argument('--repeat', type=int, default=3, help='repeat calculation of ranking functions')
    parser.add_argument('--timestamp', default='none', type=str, help='timestamp for logging naming')
    parser.add_argument('--init', default='kaiming_norm',
                        choices=['kaiming_norm', 'kaiming_norm_fanin', 'kaiming_norm_fanout'],
                        help='initialization to use')
    parser.add_argument('--super_type', type=str, default='basic', help='type of supernet: basic or nasnet-super')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
