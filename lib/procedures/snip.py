import torch
import types
from copy import deepcopy
from .slight_train import slight_train



def snip_forward_conv2d(self, x):
    return torch.nn.functional.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
    return torch.nn.functional.linear(x, self.weight * self.weight_mask, self.bias)


def snip(train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
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
    network_snip = []
    for net in networks:
        net_copy = deepcopy(net.cpu())
        net_copy.load_state_dict(deepcopy(net.state_dict()))
        net_copy = net_copy.to(device)

        for layer in net_copy.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer.weight_mask = torch.nn.Parameter(torch.ones_like(layer.weight)).to(device)
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, torch.nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

            if isinstance(layer, torch.nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # Compute gradients (but don't apply them)
        net_copy.zero_grad()
        output = net_copy(inputs)

        assert isinstance(output, tuple)
        output = output[1]

        loss = loss_func(output, targets)
        loss.backward()

        snip = 0.0
        for layer in net_copy.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if layer.weight_mask.grad is not None:
                    snip += torch.sum(torch.abs(layer.weight_mask.grad)).item()
        network_snip.append(snip)

        del net_copy
        torch.cuda.empty_cache()

    return network_snip