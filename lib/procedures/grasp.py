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

        # first pass
        net.zero_grad()
        output = net(inputs)

        assert isinstance(output, tuple)
        output = output[1]

        loss = loss_func(output, targets)
        grad_w = torch.autograd.grad(loss, weights, allow_unused=True)

        # second pass
        net.zero_grad()
        output = net(inputs)

        assert isinstance(output, tuple)
        output = output[1]

        loss = loss_func(output, targets)

        grad_f = torch.autograd.grad(loss, weights, create_graph=True, allow_unused=True)

        # accumulate gradients computed in previous step and call backwards
        z, count = 0, 0
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if grad_w[count] is not None:
                    z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

        grasp = 0.0
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if layer.weight.grad is not None:
                    grasp += torch.sum(-layer.weight.data * layer.weight.grad).item()

        network_grasp.append(grasp)

    return network_grasp