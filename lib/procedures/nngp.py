import numpy as np
import torch
from .common_methods import MetricType
from .slight_train import slight_train
from torch.nn.functional import one_hot
from itertools import islice


def get_nngp_n(train_loader, valid_loader, networks, train_mode=False, num_batch=-1, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_mode:
            network.train()
        else:
            network.eval()

    train_logits = [[] for _ in range(len(networks))]
    train_targets = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(train_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits

            train_logits[net_idx].append(logit.detach())
            train_targets[net_idx].append(targets.detach())
            torch.cuda.empty_cache()

    valid_logits = [[] for _ in range(len(networks))]
    valid_targets = [[] for _ in range(len(networks))]
    for i, (inputs, targets) in enumerate(valid_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            inputs_ = inputs.clone().cuda(device=device, non_blocking=True)
            logit = network(inputs_)
            if isinstance(logit, tuple):
                logit = logit[1]  # 201 networks: return features and logits

            valid_logits[net_idx].append(logit)
            valid_targets[net_idx].append(targets)
            torch.cuda.empty_cache()
    ######
    train_logits = [torch.concat(l, 0) for l in train_logits]
    train_targets = [torch.concat(t, 0) for t in train_targets]
    valid_logits = [torch.concat(l, 0) for l in valid_logits]
    valid_targets = [torch.concat(t, 0) for t in valid_targets]

    # one-hot labeling
    num_classes = len(valid_loader.dataset.classes)
    train_targets = [one_hot(t, num_classes=num_classes).to(torch.float32) for t in train_targets]

    train_Ks = [torch.einsum('nc,mc->nm', [l, l]) for l in train_logits]
    valid_Ks = [torch.einsum('nc,mc->nm', [l1, l2]) for l1, l2 in zip(valid_logits,train_logits)]

    acc = [-1.0 for _ in range(len(networks))]
    # Range of regularizer set manually.
    diag_reg_values = np.logspace(-7, 2, num=20)

    for net_idx in range(len(networks)):
        K_tt = train_Ks[net_idx]
        K_vt = valid_Ks[net_idx]
        labels_t = train_targets[net_idx].cuda(device=device, non_blocking=True)
        labels_v = valid_targets[net_idx].cuda(device=device, non_blocking=True)
        n_t = K_tt.shape[0]
        for epsilon in diag_reg_values:
            # Regularize K_tt.
            K_tt_reg = K_tt + epsilon * torch.trace(K_tt).cuda(device=device, non_blocking=True) / n_t * torch.eye(n_t).cuda(device=device, non_blocking=True)
            # 'try' statement, since scipty.linalg.solve can fail.
            try:
                # Perform NNGP inference to obtain validation accuracy.
                inv_labels = torch.linalg.solve(K_tt_reg, labels_t)
                # inv_labels = scipy.linalg.solve(K_tt_reg, labels_t, sym_pos=True)
                prediction = torch.matmul(K_vt, inv_labels)
                correct_predictions = (torch.argmax(prediction, dim=1) == labels_v).to(torch.float32)
                acc[net_idx] = max(acc[net_idx], torch.mean(correct_predictions).item())
            except Exception as e:
                if verbose:
                    print("Matrix inversion error for epsilon = {}, reason {}".format(epsilon, e))
                continue
    return acc


def compute_nngp_outputs(inputs, network, use_logits=False):
    with torch.no_grad():
        output = network(inputs)
        assert isinstance(output, tuple)
        if use_logits:  # 201 networks: return features and logits
            output = output[1]
        else:
            output = output[0]
        return output

def get_nngp_n_v2(train_loader, valid_loader, networks, metric=MetricType.ACC, train_mode=False, as_correlation=False, train_iters=-1, num_batch=-1, use_logits=False, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    train_logits = [[] for _ in range(len(networks))]
    train_targets = []
    for i, (inputs, targets) in enumerate(train_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            logit = compute_nngp_outputs(inputs, network, use_logits=use_logits)
            train_logits[net_idx].append(logit.detach())
            torch.cuda.empty_cache()
        train_targets.append(targets.detach())

    train_logits = [torch.concat(l, 0) for l in train_logits]
    if as_correlation:
        train_Ks = [torch.corrcoef(l) for l in train_logits]
    else:
        train_Ks = [torch.einsum('nc,mc->nm', [l, l]) for l in train_logits]


    if MetricType.require_only_matrix(metric):
        scores = []
        for k in train_Ks:
            val = metric(k)
            scores.append(val)
        return scores

    num_classes = len(valid_loader.dataset.classes)
    train_targets = torch.concat(train_targets, 0)
    train_targets = one_hot(train_targets, num_classes=num_classes).to(torch.float32).cuda(device=device, non_blocking=True)

    if metric is MetricType.LGA:
        scores = []
        for k in train_Ks:
            val = metric(k, train_targets)
            scores.append(val)
        return scores

    valid_logits = [[] for _ in range(len(networks))]
    valid_targets = []
    for i, (inputs, targets) in enumerate(valid_loader):
        if num_batch > 0 and i >= num_batch:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        for net_idx, network in enumerate(networks):
            valid_logits[net_idx].append(compute_nngp_outputs(inputs, network, use_logits=use_logits).detach())
            torch.cuda.empty_cache()
        valid_targets.append(targets)

    valid_logits = [torch.concat(l, 0) for l in valid_logits]
    valid_Ks = [torch.einsum('nc,mc->nm', [l1, l2]) for l1, l2 in zip(valid_logits, train_logits)]

    valid_targets = torch.concat(valid_targets, 0)
    valid_targets = one_hot(valid_targets, num_classes=num_classes).to(torch.float32).cuda(device=device, non_blocking=True)

    scores = [-1.0 for _ in range(len(networks))]
    # Range of regularizer set manually.
    diag_reg_values = np.logspace(-7, 2, num=20)
    for net_idx in range(len(networks)):
        K_tt = train_Ks[net_idx]
        K_vt = valid_Ks[net_idx]
        n_t = K_tt.shape[0]
        for epsilon in diag_reg_values:
            # Regularize K_tt.
            K_tt_reg = K_tt + epsilon * torch.trace(K_tt).cuda(device=device, non_blocking=True) / n_t * torch.eye(n_t).cuda(device=device, non_blocking=True)
            # 'try' statement, since scipty.linalg.solve can fail.
            try:
                # Perform NNGP inference to obtain validation accuracy.
                inv_labels = torch.linalg.solve(K_tt_reg, train_targets)
                prediction = torch.matmul(K_vt, inv_labels)

                val = metric(valid_targets, prediction)
                scores[net_idx] = max(scores[net_idx], val)
            except Exception as e:
                if verbose:
                    print("Matrix inversion error for epsilon = {}, reason {}".format(epsilon, e))
                continue
    return scores