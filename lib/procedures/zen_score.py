import torch
from .slight_train import slight_train


def zen_score(train_loader, networks, train_mode=False, train_iters=-1, mixup_gamma = 1e-2, verbose=False):
    device = torch.cuda.current_device()
    for network in networks:
        if train_iters > 0:
            slight_train(network, train_loader, train_iters, device)
        if train_mode:
            network.train()
        else:
            network.eval()

    inputs, _ = next(iter(train_loader))
    input = torch.randn(size=list(inputs.shape), device=device, dtype=inputs.dtype)
    input2 = torch.randn(size=list(inputs.shape), device=device, dtype=inputs.dtype)
    mixup_input = input + mixup_gamma * input2

    zen_scores = []
    for net in networks:
        with torch.no_grad():
            output = net(input)
            mixup_output = net(mixup_input)

            assert isinstance(output, tuple)
            assert isinstance(mixup_output, tuple)

            output = output[1]
            mixup_output = mixup_output[1]

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=1)
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            zen_scores.append(nas_score.item())
    return zen_scores