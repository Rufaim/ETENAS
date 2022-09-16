import numpy as np
import torch
from typing import Callable
from enum import Enum, auto
from .slight_train import slight_train


class RegionDivisionScoreType(Enum):
    FULL = auto()
    MEAN = auto()
    MAX = auto()


def regional_division_counter(train_loader, networks, train_mode=False, score_type=RegionDivisionScoreType.FULL, train_iters=-1, num_batch=-1, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    network_Ks = [None for _ in range(len(networks))]

    def counting_forward_hook(module_name: str, network_id: int) -> Callable:
        def fn(_, inp, __):
            try:
                if isinstance(inp, tuple):
                    inp = inp[0]
                with torch.no_grad():
                    inp = inp.view(inp.size(0), -1)
                    x = (inp > 0).float()
                    K = x @ x.t()
                    K2 = (1. - x) @ (1. - x.t())
                network_Ks[network_id] = network_Ks[network_id] + K.cpu().detach().numpy() + K2.cpu().detach().numpy()
            except Exception as e:
                if verbose:
                    print(f"Module {module_name}, in-hook exception raied, reason: {e}")
        return fn

    all_hook_handlers = []
    for net_idx, network in enumerate(networks):
        for name, module in network.named_modules():
            if isinstance(module, torch.nn.ReLU):
                all_hook_handlers.append(module.register_forward_hook(counting_forward_hook(name, net_idx)))

    network_scores = [0 for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(train_loader):
        if num_batch > 0 and i >= num_batch:
            break
        if i == 0 or score_type is not RegionDivisionScoreType.FULL:
            batch_size = inputs.shape[0]
            for j in range(len(networks)):
                network_Ks[j] = np.zeros((batch_size, batch_size))

        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network(inputs)
            if score_type is RegionDivisionScoreType.MEAN:
                network_scores[net_idx] += np.linalg.slogdet(network_Ks[net_idx])[1]
            elif score_type is RegionDivisionScoreType.MAX:
                network_scores[net_idx] = max(network_scores[net_idx], np.linalg.slogdet(network_Ks[net_idx])[1])

    if score_type is RegionDivisionScoreType.FULL:
        network_scores = [np.linalg.slogdet(k)[1] for k in network_Ks]
    elif score_type is RegionDivisionScoreType.MEAN:
        network_scores = [s/num_batch for s in network_scores]

    ## clearing
    for h in all_hook_handlers:
        h.remove()

    return network_scores
