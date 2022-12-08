import torch
from .slight_train import slight_train


@torch.no_grad()
def linearize(net):
    signs = {}
    for name, param in net.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs


# convert to orig values
@torch.no_grad()
def nonlinearize(net, signs):
    for name, param in net.state_dict().items():
        if "weight_mask" not in name:
            param.mul_(signs[name])


def synflow(train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
    return synflow_(False, train_loader, networks, train_mode, train_iters, verbose)


def logsynflow(train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
    return synflow_(True, train_loader, networks, train_mode, train_iters, verbose)


def synflow_(use_log, train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    inputs, _ = next(iter(train_loader))
    input_dim = list(inputs[0, :].shape)
    inputs = torch.ones([1] + input_dim).double().to(device)

    network_synflow = []
    for net in networks:
        # keep signs of all params
        signs = linearize(net.double())

        # Compute gradients with input of 1s
        net.zero_grad()
        net.double()
        output = net(inputs)

        assert isinstance(output, tuple)
        torch.sum(output[1]).backward()

        synflow = 0.0
        for layer in net.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if layer.weight.grad is not None:
                    g = layer.weight.grad
                    if use_log:
                        g = torch.log(torch.abs(g))
                    synflow += torch.sum(torch.abs(layer.weight * g))

        nonlinearize(net, signs)

        network_synflow.append(synflow.item())
    return network_synflow