import numpy as np
import torch
from torch.nn.functional import one_hot
from .common_methods import MetricType
from .slight_train import slight_train


def recal_bn(network, xloader, recalbn, device):
    for m in network.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    network.train()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(xloader):
            if i >= recalbn: break
            inputs = inputs.cuda(device=device, non_blocking=True)
            _, _ = network(inputs)
    return network


def get_ntk_n(xloader, networks, recalbn=0, train_mode=False, num_batch=-1):
    device = torch.cuda.current_device()
    # if recalbn > 0:
    #     network = recal_bn(network, xloader, recalbn, device)
    #     if network_2 is not None:
    #         network_2 = recal_bn(network_2, xloader, recalbn, device)
    ntks = []
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()
    ######
    grads = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(xloader):
        if num_batch > 0 and i >= num_batch: break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            network.zero_grad()
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits
            for _idx in range(len(inputs_)):
                logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
                grad = []
                for name, W in network.named_parameters():
                    if 'weight' in name and W.grad is not None:
                        grad.append(W.grad.view(-1).detach())
                grads[net_idx].append(torch.cat(grad, -1))
                network.zero_grad()
                torch.cuda.empty_cache()
    ######
    grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U')
        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0))
    return conds

def compute_ntk_grads(inputs, network):
    grads = []
    network.zero_grad()
    # inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
    logit = network(inputs)
    if isinstance(logit, tuple):
        logit = logit[1]  # 201 networks: return features and logits
    for _idx in range(len(inputs)):
        logit[_idx:_idx + 1].backward(torch.ones_like(logit[_idx:_idx + 1]), retain_graph=True)
        grad = []
        for name, W in network.named_parameters():
            if 'weight' in name and W.grad is not None:
                grad.append(W.grad.view(-1).detach())
        grads.append(torch.cat(grad, -1))
        network.zero_grad()
        torch.cuda.empty_cache()
    return grads

def get_ntk_n_v2(train_loader, valid_loader, networks, metric=MetricType.COND, train_mode=False, as_correlation=False, train_iters=-1, num_batch=-1, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    train_grads = [[] for _ in range(len(networks))]
    train_targets = []
    for i, (inputs, targets) in enumerate(train_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            train_grads[net_idx].append(compute_ntk_grads(inputs, network))
        train_targets.append(targets.detach())

    ######
    train_grads = [torch.stack(g, 0) for _grads in train_grads for g in _grads]
    if as_correlation:
        train_ntks = [torch.corrcoef(_grads) for _grads in train_grads]
    else:
        train_ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in train_grads]

    if MetricType.require_only_matrix(metric):
        scores = []
        for ntk in train_ntks:
            val = metric(ntk)
            scores.append(val)
        return scores

    num_classes = len(valid_loader.dataset.classes)
    train_targets = torch.concat(train_targets, 0)
    train_targets = one_hot(train_targets, num_classes=num_classes).to(torch.float32).cuda(device=device, non_blocking=True)

    if metric is MetricType.LGA:
        scores = []
        for ntk in train_ntks:
            val = metric(ntk, train_targets)
            scores.append(val)
        return scores

    val_grads = [[] for _ in range(len(networks))]
    val_targets = []
    for i, (inputs, targets) in enumerate(valid_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            val_grads[net_idx].append(compute_ntk_grads(inputs, network))
        val_targets.append(targets.detach())

    val_grads = [torch.stack(g, 0) for _grads in val_grads for g in _grads]
    val_ntks = [torch.einsum('nc,mc->nm', [g1, g2]) for g1,g2 in zip(val_grads,train_grads)]

    val_targets = torch.concat(val_targets, 0)
    val_targets = one_hot(val_targets, num_classes=num_classes).to(torch.float32).cuda(device=device, non_blocking=True)

    scores = [-1.0 for _ in range(len(networks))]
    for i in range(len(networks)):
        ntk_tt = train_ntks[i]
        ntk_vt = val_ntks[i]

        inv_labels = torch.linalg.solve(ntk_tt, train_targets)
        prediction = torch.matmul(ntk_vt, inv_labels)

        val = metric(val_targets, prediction)
        scores[i] = val

    return scores
