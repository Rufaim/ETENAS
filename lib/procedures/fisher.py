import torch
import types
from copy import deepcopy
from .slight_train import slight_train


def fisher_forward_conv2d(self, x):
    x = torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
    #intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self, x):
    x = torch.nn.functional.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


def fisher(train_loader, networks, train_mode=False, train_iters=-1, verbose=False):
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
    network_fisher = []
    for net in networks:
        net_copy = deepcopy(net.cpu())
        net_copy.load_state_dict(deepcopy(net.state_dict()))
        net_copy = net_copy.to(device)

        for layer in net_copy.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                # variables/op needed for fisher computation
                layer.fisher = 0.0
                layer.act = 0.0
                layer.dummy = torch.nn.Identity().to(device)

                # replace forward method of conv/linear
                if isinstance(layer, torch.nn.Conv2d):
                    layer.forward = types.MethodType(fisher_forward_conv2d, layer)
                if isinstance(layer, torch.nn.Linear):
                    layer.forward = types.MethodType(fisher_forward_linear, layer)

                # function to call during backward pass (hooked on identity op at output of layer)
                def hook_factory(layer):
                    def hook(module, grad_input, grad_output):
                        act = layer.act.detach()
                        grad = grad_output[0].detach()
                        if len(act.shape) > 2:
                            g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                        else:
                            g_nk = act * grad
                        del_k = g_nk.pow(2).mean(0).mul(0.5)
                        layer.fisher += del_k.cpu().detach()
                        del layer.act  # without deleting this, a nasty memory leak occurs! related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555

                    return hook

                # register backward hook on identity fcn to compute fisher info
                layer.dummy.register_backward_hook(hook_factory(layer))

        net_copy.zero_grad()
        output = net_copy(inputs)

        assert isinstance(output, tuple)
        output = output[1]

        loss = loss_func(output, targets)
        loss.backward()

        fisher = 0.0
        for layer in net_copy.modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                if layer.fisher is not None:
                    fisher += torch.sum(torch.abs(layer.fisher.cpu().detach()))
        network_fisher.append(fisher)

        del net_copy
        torch.cuda.empty_cache()

    return network_fisher
