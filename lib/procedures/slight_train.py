import torch


# def slight_train(network, train_loader, train_iters, device):
#     network.train()
#     opt = torch.optim.Adam(network.parameters(), lr=1e-3)
#     loss_func = torch.nn.CrossEntropyLoss()
#     for ti in range(train_iters):
#         for inputs, targets in train_loader:
#             inputs = inputs.cuda(device=device, non_blocking=True)
#             targets = targets.cuda(device=device, non_blocking=True)
#             logit = network(inputs)
#             if isinstance(logit, tuple):
#                 logit = logit[1]  # 201 networks: return features and logits
#             loss = loss_func(logit, targets)
#
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

def slight_train(network, train_loader, train_iters, device):
    network.train()
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    for i, (inputs, targets) in enumerate(train_loader):
        if i > train_iters:
            break
        inputs = inputs.cuda(device=device, non_blocking=True)
        targets = targets.cuda(device=device, non_blocking=True)
        logit = network(inputs)
        if isinstance(logit, tuple):
            logit = logit[1]  # 201 networks: return features and logits
        loss = loss_func(logit, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()