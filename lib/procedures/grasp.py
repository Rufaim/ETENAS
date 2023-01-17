import torch
from .slight_train import slight_train


def grasp(train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    inputs, targets = next(iter(train_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    network_grasp = []
    for net in networks:
        # get all applicable weights
        weights = []
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                weights.append(layer.weight)

        net.zero_grad()
        output = net(inputs)
        loss = loss_func(output, targets)
        grad_w = list(torch.autograd.grad(loss, weights, allow_unused=True))

        grasp = 0.0
        for layer in net.modules():
            if layer.weight.grad is not None:
                grasp += torch.sum(-layer.weight.data * layer.weight.grad)

        network_grasp.append(grasp)

    return network_grasp